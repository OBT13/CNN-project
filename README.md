# Arabic Handwritten Character Recognition CNN

Enhanced CNN model for recognizing 28 Arabic handwritten characters with improved accuracy (target: 80-90%).

## Features

- **Enhanced Architecture**: Deep CNN with residual connections
- **Proper Data Augmentation**: Arabic-character-appropriate augmentations
- **Validation & Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best model automatically
- **Comprehensive Training**: Gradient clipping, learning rate scheduling

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision pandas matplotlib numpy Pillow
```

## Usage

### Training the Model

Run the training script:
```bash
python train.py
```

The script will:
- Load and split the training data (80% train, 20% validation)
- Train the model with early stopping
- Save the best model as `arabic_cnn_gpu.pth`
- Evaluate on test set and display results
- Generate error analysis visualization

**Training Output:**
- Real-time training/validation loss and accuracy
- Learning rate adjustments
- Best model checkpointing
- Final test accuracy

### Making Predictions

1. Place your test image as `test.png` in the project directory
2. Run:
```bash
python predict.py
```

The script will output the predicted Arabic character and confidence score.

## Model Architecture

- **Input**: 32x32 grayscale images
- **Architecture**: 
  - Initial conv layer (1→32 channels)
  - 3 residual blocks with increasing channels (32→64→128→256)
  - Global Average Pooling
  - Fully connected classifier (256→512→256→28)
- **Regularization**: BatchNorm, Dropout, Weight Decay

## Data Structure

Expected data structure:
```
data/
  ├── csvTrainImages 13440x1024.csv
  ├── csvTrainLabel 13440x1.csv
  ├── csvTestImages 3360x1024.csv
  └── csvTestLabel 3360x1.csv
```

## Training Parameters

- **Batch Size**: 128
- **Learning Rate**: 0.0005 (with ReduceLROnPlateau scheduling)
- **Optimizer**: AdamW with weight decay (1e-4)
- **Early Stopping**: Patience of 10 epochs
- **Max Epochs**: 100 (stops early if no improvement)

## Expected Results

With the improvements, the model should achieve:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Test Accuracy**: 75-90%

## Files

- `train.py`: Main training script with validation and early stopping
- `model.py`: Enhanced CNN architecture with residual blocks
- `dataset.py`: Dataset loader for Arabic character images
- `predict.py`: Inference script for single image prediction
