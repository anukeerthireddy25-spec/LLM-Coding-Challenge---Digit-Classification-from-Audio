import torch
import sounddevice as sd
import numpy as np
import time
from collections import deque

from src.model import LightweightCNN
from src.preprocess import standardize_audio, to_mel_spectrogram, SAMPLE_RATE, TARGET_LENGTH_SAMPLES

# --- Configuration ---
MODEL_PATH = "saved_models/best_model.pth"
DEVICE = "cpu"  # Real-time apps usually run better on CPU for low latency
BUFFER_DURATION_S = 1.5  # How much audio to keep in the buffer
ANALYSIS_WINDOW_SAMPLES = TARGET_LENGTH_SAMPLES # ~1 sec, must match training
ACTIVATION_THRESHOLD = 0.01 # RMS energy threshold to start predicting

# --- Global Variables ---
audio_buffer = deque(maxlen=int(BUFFER_DURATION_S * SAMPLE_RATE))
last_prediction_time = 0
prediction_cooldown_s = 1.0 # Wait 1 second between predictions

# Load the trained model
print("Loading model...")
model = LightweightCNN(num_classes=10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded. Ready for real-time prediction.")

def audio_callback(indata, frames, time_info, status):
    """This function is called for each new audio chunk from the microphone."""
    global audio_buffer, last_prediction_time
    if status:
        print(status)
    
    # Add new audio data to our buffer
    audio_buffer.extend(indata[:, 0])

    # Check if enough time has passed since the last prediction
    if time.time() - last_prediction_time < prediction_cooldown_s:
        return

    # Check if we have enough data for a full analysis window
    if len(audio_buffer) >= ANALYSIS_WINDOW_SAMPLES:
        # Get the most recent data for analysis
        analysis_window = np.array(list(audio_buffer)[-ANALYSIS_WINDOW_SAMPLES:])
        
        # --- Energy-based Activation ---
        # Calculate Root Mean Square (RMS) energy to detect speech
        rms_energy = np.sqrt(np.mean(analysis_window**2))
        
        if rms_energy > ACTIVATION_THRESHOLD:
            print(f"Speech detected (RMS: {rms_energy:.4f})... ", end="", flush=True)
            
            # --- Prediction Pipeline ---
            waveform = torch.tensor(analysis_window, dtype=torch.float32).unsqueeze(0)
            
            # Preprocess and predict
            with torch.no_grad():
                # Note: Standardization is already part of the pipeline inside predict
                waveform = standardize_audio(waveform)
                mel_spectrogram = to_mel_spectrogram(waveform).unsqueeze(0).to(DEVICE)
                prediction = model(mel_spectrogram)
                predicted_index = torch.argmax(prediction, dim=1).item()

            print(f"Predicted Digit: {predicted_index}")
            
            # Update last prediction time to enforce cooldown
            last_prediction_time = time.time()
            # Optional: Clear buffer after prediction to avoid re-predicting the same sound
            # audio_buffer.clear()


def main():
    print("\n--- Real-Time Spoken Digit Classification ---")
    print(f"Listening with a {int(ANALYSIS_WINDOW_SAMPLES/SAMPLE_RATE * 1000)}ms analysis window.")
    print(f"Activation Threshold (RMS): {ACTIVATION_THRESHOLD}")
    print("Speak a digit into your microphone...")
    print("Press Ctrl+C to exit.")

    try:
        # Start the microphone input stream
        with sd.InputStream(callback=audio_callback,
                             channels=1,
                             samplerate=SAMPLE_RATE,
                             dtype='float32'):
            while True:
                # Keep the script running
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
