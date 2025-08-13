import torch
import torchaudio
import numpy as np

# --- Configuration ---
# Using the values specified in the technical guide
SAMPLE_RATE = 8000
TARGET_LENGTH_SAMPLES = 8192  # ~1.024 seconds

# --- Main Preprocessing Functions ---

def standardize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """
    Standardizes a single audio waveform to a fixed length and amplitude range.

    This function performs two key steps as recommended in the guide:
    1.  Length Standardization: Pads or truncates the audio to TARGET_LENGTH_SAMPLES.
    2.  Amplitude Normalization: Scales the waveform to a [-1, 1] range.

    Args:
        waveform (torch.Tensor): The input audio waveform.

    Returns:
        torch.Tensor: The standardized audio waveform.
    """
    num_channels, current_len = waveform.shape

    # 1. Standardize Length
    if current_len > TARGET_LENGTH_SAMPLES:
        # Truncate if longer
        waveform = waveform[:, :TARGET_LENGTH_SAMPLES]
    elif current_len < TARGET_LENGTH_SAMPLES:
        # Pad with zeros if shorter
        padding_needed = TARGET_LENGTH_SAMPLES - current_len
        # Symmetrical padding (pad on both sides)
        waveform = torch.nn.functional.pad(waveform, (padding_needed // 2, padding_needed - padding_needed // 2))

    # 2. Normalize Amplitude
    # This prevents the model from being influenced by volume differences.
    max_abs_val = torch.max(torch.abs(waveform))
    if max_abs_val > 0:
        waveform = waveform / max_abs_val

    return waveform

def to_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """
    Converts a standardized audio waveform into a Mel spectrogram.

    This is the chosen feature representation, ideal for a 2D CNN as it
    treats the time-frequency data like an image.

    Args:
        waveform (torch.Tensor): The standardized input waveform.

    Returns:
        torch.Tensor: The resulting Mel spectrogram.
    """
    # Parameters chosen based on common values for speech processing
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=400,       # Window size for Fourier Transform
        hop_length=160,  # Step size between windows
        n_mels=64        # Number of Mel frequency bands
    )
    mel_spectrogram = mel_transform(waveform)

    # Apply log scaling for better dynamic range
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

    return log_mel_spectrogram

def apply_augmentations(mel_spectrogram: torch.Tensor) -> torch.Tensor:
    """
    Applies SpecAugment to a Mel spectrogram for data augmentation.

    This function is only used during training to make the model more robust.
    It randomly masks frequency and time bands, forcing the model to learn
    more distributed features.

    Args:
        mel_spectrogram (torch.Tensor): The input Mel spectrogram.

    Returns:
        torch.Tensor: The augmented Mel spectrogram.
    """
    # Frequency Masking: Randomly masks a horizontal band of frequencies.
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
    mel_spectrogram = freq_mask(mel_spectrogram)

    # Time Masking: Randomly masks a vertical band of time steps.
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
    mel_spectrogram = time_mask(mel_spectrogram)

    return mel_spectrogram