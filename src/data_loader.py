import torch
import torchaudio
from torch.utils.data import Dataset
import os
import glob
import urllib.request
import zipfile

class SpokenDigitDataset(Dataset):
    """
    PyTorch Dataset for the Free Spoken Digit Dataset (FSDD).

    This class uses a robust method of downloading the dataset directly from its
    source URL using standard Python libraries to avoid versioning issues.
    It then manually applies the official train/test split based on the filename.
    """
    _URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.9.zip"

    def __init__(self, split="train", apply_augmentation=False):
        """
        Args:
            split (str): "train" or "test" to specify the dataset split.
            apply_augmentation (bool): Whether to apply SpecAugment. Should be True
                                       only for the training set.
        """
        if split not in ["train", "test"]:
            raise ValueError("Split must be 'train' or 'test'")

        self.apply_augmentation = apply_augmentation
        self.split = split
        
        # --- Download and extract the dataset using standard libraries ---
        root_dir = "./data"
        os.makedirs(root_dir, exist_ok=True)
        
        archive_path = os.path.join(root_dir, "fsdd.zip")
        extracted_dir = os.path.join(root_dir, "free-spoken-digit-dataset-1.0.9")

        if not os.path.isdir(extracted_dir):
            print(f"Downloading dataset from {self._URL}...")
            urllib.request.urlretrieve(self._URL, archive_path)
            print("Download complete. Extracting...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(root_dir)
            print("Extraction complete.")

        # --- Find all audio files and create the train/test split ---
        wav_files = glob.glob(os.path.join(extracted_dir, "recordings", "*.wav"))
        
        self.file_list = []
        for file_path in wav_files:
            filename = os.path.basename(file_path)
            # Filename format: {digitLabel}_{speakerName}_{index}.wav
            parts = filename.split('_')
            label = int(parts[0])
            index = int(parts[-1].split('.')[0])
            
            is_test_sample = index <= 4

            if split == "train" and not is_test_sample:
                self.file_list.append({"path": file_path, "label": label})
            elif split == "test" and is_test_sample:
                self.file_list.append({"path": file_path, "label": label})

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Fetches a single data point and processes it.
        """
        item = self.file_list[idx]
        
        # Load audio from the file path
        waveform, _ = torchaudio.load(item['path'])
        label = torch.tensor(item['label'], dtype=torch.long)

        # Apply the preprocessing pipeline
        from src.preprocess import standardize_audio, to_mel_spectrogram, apply_augmentations
        waveform = standardize_audio(waveform)
        mel_spectrogram = to_mel_spectrogram(waveform)

        # Apply augmentations only if specified (i.e., for the training set)
        if self.apply_augmentation:
            mel_spectrogram = apply_augmentations(mel_spectrogram)

        return mel_spectrogram, label
