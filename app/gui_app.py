import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import torch
import queue
import threading
import yaml
from collections import deque
import time

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- This is a GUI application for real-time digit classification ---

# --- Configuration and Model Loading ---
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found.")
    exit()

MODEL_PATH = f"{config['model']['saved_models_dir']}/{config['model']['best_model_name']}"
DEVICE = "cpu"
SAMPLE_RATE = config['data']['sample_rate']
TARGET_LENGTH_SAMPLES = config['data']['target_length_samples']
ACTIVATION_THRESHOLD = config['realtime_app']['activation_threshold']
PREDICTION_COOLDOWN_S = config['realtime_app']['prediction_cooldown_s']

try:
    from src.model import LightweightCNN
    from src.preprocess import standardize_audio, to_mel_spectrogram
    model = LightweightCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


class SpokenDigitGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spoken Digit Classifier")
        self.geometry("700x600")

        self.audio_queue = queue.Queue()
        self.stream = None
        self.is_listening = False
        self.last_prediction_time = 0
        
        # --- MODIFIED: State variables for new "trigger and capture" logic ---
        self.is_capturing = False
        self.capture_buffer = []

        five_seconds_length = 5 * SAMPLE_RATE
        self.session_recording_buffer = deque(maxlen=five_seconds_length)
        self.last_recorded_clip = None

        self._setup_ui()
        self.process_queue()

    def _setup_ui(self):
        # --- Main Frames ---
        control_frame = ttk.Frame(self, padding="10")
        control_frame.pack(side="top", fill="x")

        debug_frame = ttk.Frame(self, padding="10")
        debug_frame.pack(side="top", fill="x")

        display_frame = ttk.Frame(self, padding="10")
        display_frame.pack(side="top", fill="both", expand=True)
        
        # --- Controls ---
        self.listen_button = ttk.Button(control_frame, text="Start Listening", command=self.toggle_listening)
        self.listen_button.pack(side="left")

        ttk.Label(control_frame, text="Microphone:").pack(side="left", padx=(20, 5))
        self.mic_variable = tk.StringVar()
        self.mic_options = self.get_mic_options()
        self.mic_dropdown = ttk.Combobox(control_frame, textvariable=self.mic_variable, values=self.mic_options, state="readonly", width=40)
        if self.mic_options:
            self.mic_dropdown.current(0)
        self.mic_dropdown.pack(side="left")

        self.status_label = ttk.Label(control_frame, text="Status: Idle")
        self.status_label.pack(side="left", padx=10)

        # --- Debug Controls ---
        ttk.Label(debug_frame, text="Debugging Tools:").pack(side="left")
        self.play_button = ttk.Button(debug_frame, text="Play Last 5 Seconds", command=self.play_clip, state="disabled")
        self.play_button.pack(side="left")

        self.rms_label = ttk.Label(debug_frame, text=f"Live RMS: 0.0000 (Threshold: {ACTIVATION_THRESHOLD})")
        self.rms_label.pack(side="left", padx=20)


        # --- Display Area ---
        prediction_frame = ttk.Frame(display_frame)
        prediction_frame.pack(side="top", fill="x", pady=10)

        plot_frame = ttk.Frame(display_frame)
        plot_frame.pack(side="bottom", fill="both", expand=True)

        ttk.Label(prediction_frame, text="Predicted Digit:", font=("Helvetica", 16)).pack(side="left")
        self.prediction_label = ttk.Label(prediction_frame, text="--", font=("Helvetica", 48, "bold"))
        self.prediction_label.pack(side="left", padx=20)

        # --- Matplotlib Plots ---
        self.fig_waveform = Figure(figsize=(8, 2), dpi=100)
        self.ax_waveform = self.fig_waveform.add_subplot(111)
        self.ax_waveform.set_title("Live Audio Waveform")
        self.ax_waveform.set_ylim(-1, 1)
        self.waveform_canvas = FigureCanvasTkAgg(self.fig_waveform, master=plot_frame)
        self.waveform_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.fig_probs = Figure(figsize=(8, 2), dpi=100)
        self.ax_probs = self.fig_probs.add_subplot(111)
        self.ax_probs.set_title("Prediction Confidence")
        self.ax_probs.set_xticks(range(10))
        self.ax_probs.set_ylim(0, 1)
        self.probs_canvas = FigureCanvasTkAgg(self.fig_probs, master=plot_frame)
        self.probs_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def get_mic_options(self):
        try:
            devices = sd.query_devices()
            self.input_devices = {f"{i}: {d['name']}": i for i, d in enumerate(devices) if d['max_input_channels'] > 0}
            return list(self.input_devices.keys())
        except Exception as e:
            print(f"Could not query audio devices: {e}")
            return []

    def play_clip(self):
        if self.last_recorded_clip is not None and len(self.last_recorded_clip) > 0:
            try:
                self.status_label.config(text="Status: Playing...")
                self.update()
                sd.play(self.last_recorded_clip, samplerate=SAMPLE_RATE)
                sd.wait()
                self.status_label.config(text="Status: Playback finished.")
            except Exception as e:
                self.status_label.config(text=f"Error: {e}")


    def toggle_listening(self):
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self):
        self.session_recording_buffer.clear()
        self.play_button.config(state="disabled")
        
        self.mic_dropdown.config(state="disabled")
        self.is_listening = True
        self.listen_button.config(text="Stop Listening")
        self.status_label.config(text="Status: Listening...")
        
        selected_mic_name = self.mic_variable.get()
        device_id = self.input_devices.get(selected_mic_name)

        def audio_callback(indata, frames, time_info, status):
            self.audio_queue.put(indata.copy())

        try:
            self.stream = sd.InputStream(
                device=device_id,
                callback=audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype='float32'
            )
            self.stream.start()
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.stop_listening()

    def stop_listening(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        if self.session_recording_buffer:
            self.last_recorded_clip = np.array(self.session_recording_buffer)
            self.play_button.config(state="normal")
            
        self.is_listening = False
        self.listen_button.config(text="Start Listening")
        self.status_label.config(text="Status: Idle")
        self.mic_dropdown.config(state="readonly")


    def process_queue(self):
        try:
            audio_chunk = self.audio_queue.get_nowait()
            
            chunk_1d = audio_chunk[:, 0]
            if self.is_listening:
                self.session_recording_buffer.extend(chunk_1d)
            
            self.ax_waveform.clear()
            self.ax_waveform.set_ylim(-1, 1)
            self.ax_waveform.plot(chunk_1d)
            self.waveform_canvas.draw()
            
            rms_energy = np.sqrt(np.mean(chunk_1d**2))
            self.rms_label.config(text=f"Live RMS: {rms_energy:.4f} (Threshold: {ACTIVATION_THRESHOLD})")

            # --- REVISED PREDICTION LOGIC ---
            
            # 1. Trigger on a loud sound if not already capturing
            if rms_energy > ACTIVATION_THRESHOLD and not self.is_capturing:
                if time.time() - self.last_prediction_time > PREDICTION_COOLDOWN_S:
                    self.is_capturing = True
                    self.capture_buffer = [] # Start with a fresh buffer
                    self.status_label.config(text="Status: Capturing...")

            # 2. If we are in the capturing state, add audio to the capture buffer
            if self.is_capturing:
                self.capture_buffer.extend(chunk_1d)
                
                # 3. Once enough audio is captured, predict
                if len(self.capture_buffer) >= TARGET_LENGTH_SAMPLES:
                    full_window = np.array(self.capture_buffer[:TARGET_LENGTH_SAMPLES])
                    waveform = torch.tensor(full_window, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        waveform = standardize_audio(waveform)
                        mel_spectrogram = to_mel_spectrogram(waveform).unsqueeze(0).to(DEVICE)
                        prediction_logits = model(mel_spectrogram)
                        probabilities = torch.exp(prediction_logits).squeeze().numpy()
                        predicted_index = np.argmax(probabilities)
                    
                    self.prediction_label.config(text=str(predicted_index))
                    self.update_probs_plot(probabilities)
                    
                    # 4. Reset for the next prediction
                    self.is_capturing = False
                    self.last_prediction_time = time.time()
                    self.status_label.config(text="Status: Listening...")

        except queue.Empty:
            pass
        finally:
            self.after(50, self.process_queue)

    def update_probs_plot(self, probabilities):
        self.ax_probs.clear()
        self.ax_probs.set_ylim(0, 1)
        self.ax_probs.set_xticks(range(10))
        self.ax_probs.bar(range(10), probabilities, color='skyblue')
        self.probs_canvas.draw()

    def on_closing(self):
        self.stop_listening()
        self.destroy()

if __name__ == "__main__":
    app = SpokenDigitGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
