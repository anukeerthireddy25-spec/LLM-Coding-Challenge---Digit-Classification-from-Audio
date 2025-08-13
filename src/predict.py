import torch
import torchaudio
import argparse

from src.model import LightweightCNN
from src.preprocess import standardize_audio, to_mel_spectrogram

def predict(model, waveform, device):
    """Runs prediction on a single audio waveform."""
    model.eval()
    waveform = waveform.to(device)
    
    # Apply the same preprocessing as training data (without augmentation)
    waveform = standardize_audio(waveform)
    mel_spectrogram = to_mel_spectrogram(waveform)
    mel_spectrogram = mel_spectrogram.unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        prediction = model(mel_spectrogram)
        predicted_index = torch.argmax(prediction, dim=1).item()
    
    return predicted_index

def main():
    parser = argparse.ArgumentParser(description="Predict a spoken digit from a WAV file.")
    parser.add_argument("--file", type=str, required=True, help="Path to the WAV audio file.")
    parser.add_argument("--model_path", type=str, default="saved_models/best_model.pth", help="Path to the trained model checkpoint.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the trained model
    model = LightweightCNN(num_classes=10)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Load the audio file
    try:
        waveform, sample_rate = torchaudio.load(args.file)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Resample if necessary
    if sample_rate != 8000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
        waveform = resampler(waveform)

    # Run prediction
    predicted_digit = predict(model, waveform, device)
    print(f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    main()
