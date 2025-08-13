import torch
import sounddevice as sd
import numpy as np
import time
from collections import deque
import yaml

# --- This is an interactive debug version of the script ---

def select_device():
    """Prints all available audio devices and asks the user to select one."""
    print("\n--- Available Audio Input Devices ---")
    try:
        devices = sd.query_devices()
        input_devices = [d for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                 # Mark the default device with an asterisk
                is_default = (i == sd.default.device[0])
                prefix = "> " if is_default else "  "
                print(f"{prefix}ID {i}: {device['name']}")

        if not input_devices:
            print("\nError: No input devices found.")
            print("Please ensure you have a microphone connected and drivers are installed.")
            return None

        print("---------------------------------\n")
        while True:
            try:
                device_id = int(input("Please enter the ID of the microphone you want to use: "))
                if devices[device_id]['max_input_channels'] > 0:
                    return device_id
                else:
                    print("Invalid ID. Please choose an ID from the list of input devices.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid number ID.")

    except Exception as e:
        print(f"Could not query audio devices: {e}")
        return None

# --- Configuration Loading ---
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found. Please ensure the file exists in the root directory.")
    exit()

# --- Constants from Config ---
MODEL_PATH = f"{config['model']['saved_models_dir']}/{config['model']['best_model_name']}"
DEVICE = "cpu"
SAMPLE_RATE = config['data']['sample_rate']
TARGET_LENGTH_SAMPLES = config['data']['target_length_samples']
BUFFER_DURATION_S = config['realtime_app']['buffer_duration_s']
ACTIVATION_THRESHOLD = config['realtime_app']['activation_threshold']
PREDICTION_COOLDOWN_S = config['realtime_app']['prediction_cooldown_s']

# --- Global Variables ---
audio_buffer = deque(maxlen=int(BUFFER_DURATION_S * SAMPLE_RATE))
last_prediction_time = 0

# --- Model Loading ---
try:
    from src.model import LightweightCNN
    from src.preprocess import standardize_audio, to_mel_spectrogram
    print("Loading model...")
    model = LightweightCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded. Ready for real-time prediction.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please run the training script (src/train.py) first to generate the model file.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

def audio_callback(indata, frames, time_info, status):
    """This function is called for each new audio chunk from the microphone."""
    global audio_buffer, last_prediction_time
    if status:
        print(status, flush=True)
    
    audio_buffer.extend(indata[:, 0])
    
    # --- DEBUG PRINT: Continuously print the current RMS energy ---
    current_rms = np.sqrt(np.mean(indata[:, 0]**2))
    print(f"Live RMS: {current_rms:.4f}", end='\r', flush=True)

    if time.time() - last_prediction_time < PREDICTION_COOLDOWN_S:
        return

    if len(audio_buffer) >= TARGET_LENGTH_SAMPLES:
        analysis_window = np.array(list(audio_buffer)[-TARGET_LENGTH_SAMPLES:])
        rms_energy = np.sqrt(np.mean(analysis_window**2))
        
        if rms_energy > ACTIVATION_THRESHOLD:
            print(" " * 50, end='\r') 
            print(f"Speech detected (RMS: {rms_energy:.4f})... ", end="", flush=True)
            
            waveform = torch.tensor(analysis_window, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                waveform = standardize_audio(waveform)
                mel_spectrogram = to_mel_spectrogram(waveform).unsqueeze(0).to(DEVICE)
                prediction = model(mel_spectrogram)
                predicted_index = torch.argmax(prediction, dim=1).item()

            print(f"Predicted Digit: {predicted_index}")
            last_prediction_time = time.time()

def main():
    device_id = select_device()
    if device_id is None:
        return

    print("\n--- Real-Time Spoken Digit Classification ---")
    print(f"Listening on device ID {device_id}...")
    print(f"Activation Threshold is set to: {ACTIVATION_THRESHOLD}")
    print("Speak a digit. Watch the 'Live RMS' value change.")
    print("Press Ctrl+C to exit.")

    try:
        with sd.InputStream(callback=audio_callback,
                             device=device_id,
                             channels=1,
                             samplerate=SAMPLE_RATE,
                             dtype='float32'):
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("This might be an issue with your microphone's sample rate or drivers.")

if __name__ == "__main__":
    main()
