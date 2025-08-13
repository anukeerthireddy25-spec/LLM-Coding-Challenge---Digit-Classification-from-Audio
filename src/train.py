import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from src.data_loader import SpokenDigitDataset
from src.model import LightweightCNN
from src.preprocess import SAMPLE_RATE # For inference time calculation

# --- Configuration ---
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "saved_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    """Performs one full training pass over the dataset."""
    model.train() # Set model to training mode
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def validate(model, data_loader, loss_fn, device):
    """Performs one full validation pass."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_targets = []

    with torch.no_grad(): # No need to calculate gradients for validation
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted_indices = torch.max(predictions, 1)
            correct_predictions += (predicted_indices == targets).sum().item()
            
            all_preds.extend(predicted_indices.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct_predictions / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy, all_preds, all_targets

def main():
    """Main function to orchestrate the training and evaluation process."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    print("Loading datasets...")
    # Training dataset with augmentations
    train_dataset = SpokenDigitDataset(split="train", apply_augmentation=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Test dataset without augmentations
    test_dataset = SpokenDigitDataset(split="test", apply_augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 2. Initialize Model, Loss, and Optimizer ---
    model = LightweightCNN(num_classes=10).to(device)
    loss_fn = nn.NLLLoss() # As specified in the guide for LogSoftmax output
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    print("Starting training...")
    best_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy, _, _ = validate(model, test_loader, loss_fn, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> "
              f"Train Loss: {train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_model.pth"))
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")

    # --- 4. Final Evaluation ---
    print("\nTraining finished. Evaluating best model on the test set.")
    # Load the best performing model
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "best_model.pth")))
    
    test_loss, test_accuracy, all_preds, all_targets = validate(model, test_loader, loss_fn, device)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    # --- 5. Confusion Matrix ---
    print("Generating confusion matrix...")
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")
    
    # --- 6. Inference Time ---
    print("Calculating average inference time...")
    model.to("cpu") # Test on CPU as per guide's recommendation
    model.eval()
    
    # Get a single sample
    dummy_input, _ = test_dataset[0]
    dummy_input = dummy_input.unsqueeze(0) # Add batch dimension
    
    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)
        
    # Measure
    num_inferences = 100
    start_time = time.time()
    for _ in range(num_inferences):
        with torch.no_grad():
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_inference_time_ms = ((end_time - start_time) / num_inferences) * 1000
    print(f"Average Inference Time on CPU: {avg_inference_time_ms:.2f} ms")


if __name__ == "__main__":
    main()

