import torch
import torch.nn as nn
import yaml

class LightweightCNN(nn.Module):
    """
    A lightweight 2D Convolutional Neural Network for audio classification.

    This architecture is designed to be efficient and effective for classifying
    spoken digits from Mel spectrograms, as recommended in the technical guide.
    """
    def __init__(self, num_classes=None, dropout_rate=None):
        """
        Initializes the model.
        Args:
            num_classes (int, optional): Number of output classes.
            dropout_rate (float, optional): Dropout probability.
        """
        super(LightweightCNN, self).__init__()

        # --- Load configuration safely within the init method ---
        if num_classes is None or dropout_rate is None:
            loaded_config = None
            try:
                with open("config.yaml", 'r') as f:
                    # Check if file is not empty before loading
                    if f.read().strip():
                        f.seek(0) # Reset file pointer after reading
                        loaded_config = yaml.safe_load(f)
            except FileNotFoundError:
                loaded_config = None

            # Safely get model config, falling back to defaults
            model_config = loaded_config.get('model', {}) if loaded_config else {}
            
            if num_classes is None:
                num_classes = model_config.get('num_classes', 10)
            if dropout_rate is None:
                dropout_rate = model_config.get('dropout_rate', 0.5)

        # --- Convolutional Blocks ---
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # --- Classifier Head ---
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(3072, 128), 
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, n_mels, n_frames)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x = self.flatten(x)
        logits = self.classifier(x)
        
        # Apply LogSoftmax to get log-probabilities for NLLLoss
        output = self.log_softmax(logits)
        
        return output

