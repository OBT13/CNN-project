"""
Arabic Handwritten Character Recognition - Training Script

This script trains a deep CNN model to recognize 28 Arabic handwritten characters.
It includes data augmentation, validation splitting, early stopping, and comprehensive
training metrics tracking.

Key Features:
    - Proper data augmentation for Arabic characters (no inappropriate flips)
    - Train/validation split for overfitting prevention
    - Early stopping based on validation accuracy
    - Learning rate scheduling
    - Gradient clipping for training stability
    - Model checkpointing (saves best model)
    - Comprehensive evaluation on test set
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import StrongCNN
from dataset import ArabicDataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset (validation or test set).
    
    Computes average loss and accuracy by running inference on all batches
    in the data loader. Sets model to evaluation mode (disables dropout, etc.)
    and disables gradient computation for efficiency.
    
    Args:
        model (torch.nn.Module): The CNN model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader for the dataset
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss)
        device (str): Device to run evaluation on ('cuda' or 'cpu')
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
            - average_loss (float): Average loss across all samples
            - accuracy_percentage (float): Classification accuracy as percentage (0-100)
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient computation for efficiency during evaluation
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy


# ============================================================================
# 1. DATA AUGMENTATION AND PREPROCESSING
# ============================================================================

# Training transforms: Apply augmentation to increase dataset diversity
# Note: We avoid horizontal/vertical flips as they are inappropriate for Arabic text
train_transform = transforms.Compose([
    transforms.RandomRotation(15),  # Rotate images by up to ±15 degrees
    transforms.RandomAffine(
        degrees=5,                  # Small rotation variations
        translate=(0.1, 0.1),       # Random translation up to 10% of image size
        scale=(0.9, 1.1)            # Random scaling between 90% and 110%
    ),
    transforms.ToTensor(),          # Convert PIL Image to tensor and scale to [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range
])

# Test transforms: Minimal preprocessing (no augmentation)
# Test data should be evaluated without augmentation for accurate metrics
test_transform = transforms.Compose([
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Same normalization as training
])


# ============================================================================
# 2. DEVICE SETUP
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)


# ============================================================================
# 3. DATA LOADING AND SPLITTING
# ============================================================================

# Load full training dataset with augmentation transforms
full_train_dataset = ArabicDataset(
    "data/csvTrainImages 13440x1024.csv",
    "data/csvTrainLabel 13440x1.csv",
    transform=train_transform
)

# Create validation dataset with test transforms (no augmentation for validation)
full_val_dataset = ArabicDataset(
    "data/csvTrainImages 13440x1024.csv",
    "data/csvTrainLabel 13440x1.csv",
    transform=test_transform
)

# Split dataset into training (80%) and validation (20%) sets
# Using same random seed ensures train/val split is consistent
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
torch.manual_seed(42)  # Set random seed for reproducibility
train_dataset, _ = random_split(full_train_dataset, [train_size, val_size])
_, val_dataset = random_split(full_val_dataset, [train_size, val_size])


# ============================================================================
# 4. DATA LOADERS
# ============================================================================

# Training data loader: shuffles data and loads in batches
train_loader = DataLoader(
    train_dataset,
    batch_size=128,      # Number of samples per batch
    shuffle=True,        # Shuffle data each epoch
    num_workers=0        # Number of subprocesses for data loading
)

# Validation data loader: no shuffling needed for validation
val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,       # Don't shuffle validation data
    num_workers=0
)


# ============================================================================
# 5. MODEL INITIALIZATION
# ============================================================================

model = StrongCNN().to(device)
criterion = nn.CrossEntropyLoss()  # Standard loss for multi-class classification

# AdamW optimizer with weight decay for regularization
# Lower learning rate (0.0005) for more stable training
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)


# ============================================================================
# 6. LEARNING RATE SCHEDULER
# ============================================================================

# Reduce learning rate when validation accuracy plateaus
# This helps fine-tune the model when it stops improving
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',          # Monitor validation accuracy (maximize)
    factor=0.5,          # Reduce LR by 50% when plateau detected
    patience=5           # Wait 5 epochs before reducing LR
)


# ============================================================================
# 7. TRAINING CONFIGURATION
# ============================================================================

# Early stopping: stop training if validation accuracy doesn't improve
best_val_accuracy = 0.0
patience = 10            # Number of epochs to wait before stopping
patience_counter = 0

# Track training metrics for analysis
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


# ============================================================================
# 8. TRAINING LOOP
# ============================================================================

epochs = 100  # Maximum number of epochs (early stopping may stop earlier)
print("Starting training...")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

for epoch in range(epochs):
    # ========================================================================
    # TRAINING PHASE
    # ========================================================================
    model.train()  # Set model to training mode (enables dropout, etc.)
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # Iterate through training batches
    for images, labels in train_loader:
        # Move data to appropriate device (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass: compute predictions
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        
        # Gradient clipping: prevent exploding gradients
        # Clips gradients to have max norm of 1.0 for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()  # Update model parameters

        # Track training metrics
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted class indices
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # Calculate training accuracy and average loss
    train_accuracy = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # ========================================================================
    # VALIDATION PHASE
    # ========================================================================
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    
    # ========================================================================
    # LEARNING RATE SCHEDULING
    # ========================================================================
    # Adjust learning rate based on validation accuracy
    scheduler.step(val_accuracy)
    
    # ========================================================================
    # METRICS TRACKING
    # ========================================================================
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    # ========================================================================
    # PROGRESS REPORTING
    # ========================================================================
    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ========================================================================
    # MODEL CHECKPOINTING
    # ========================================================================
    # Save model if validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "arabic_cnn_gpu.pth")
        print(f"  ✓ Best model saved! (Val Acc: {best_val_accuracy:.2f}%)")
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1
    
    # ========================================================================
    # EARLY STOPPING
    # ========================================================================
    # Stop training if validation accuracy hasn't improved for 'patience' epochs
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        break

print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.2f}%")


# ============================================================================
# 9. TEST SET EVALUATION
# ============================================================================

print("\n" + "="*50)
print("Evaluating on test set...")
print("="*50)

# Load the best model (highest validation accuracy)
model.load_state_dict(torch.load("arabic_cnn_gpu.pth"))

# Load test dataset
test_dataset = ArabicDataset(
    "data/csvTestImages 3360x1024.csv",
    "data/csvTestLabel 3360x1.csv",
    transform=test_transform
)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Evaluate on test set
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
print(f'\nModel saved as: arabic_cnn_gpu.pth')
print('Training and evaluation completed successfully!')
