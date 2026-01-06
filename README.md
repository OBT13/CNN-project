# Arabic Handwritten Character Recognition using Deep CNN

A comprehensive deep learning project for recognizing 28 Arabic handwritten characters using a Convolutional Neural Network (CNN) with residual connections. This project demonstrates modern deep learning practices including data augmentation, validation strategies, early stopping, and model optimization techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture Details](#architecture-details)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Understanding the Code](#understanding-the-code)
- [Training Process Explained](#training-process-explained)
- [Model Architecture Explained](#model-architecture-explained)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements a deep CNN model that can recognize 28 Arabic handwritten characters (ا through ي) from 32x32 grayscale images. The model achieves high accuracy (target: 80-90%) through:

- **Enhanced Architecture**: Deep network with residual connections for better gradient flow
- **Smart Data Augmentation**: Augmentations appropriate for Arabic text (no inappropriate flips)
- **Validation Strategy**: Train/validation split to prevent overfitting
- **Early Stopping**: Automatically stops training when model stops improving
- **Advanced Training**: Learning rate scheduling, gradient clipping, and regularization

---

## 📁 Project Structure

```
CNN-project/
├── data/                                    # Dataset directory
│   ├── csvTrainImages 13440x1024.csv      # Training images (flattened)
│   ├── csvTrainLabel 13440x1.csv          # Training labels
│   ├── csvTestImages 3360x1024.csv        # Test images
│   └── csvTestLabel 3360x1.csv            # Test labels
├── model.py                                # CNN model architecture
├── dataset.py                              # Dataset loader class
├── train.py                                # Training script
├── predict.py                              # Inference script
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── .gitignore                              # Git ignore rules
```

---

## 🏗️ Architecture Details

### Model Architecture Flow

```
Input (32x32 grayscale)
    ↓
Initial Conv: 1 → 32 channels (32x32)
    ↓
Residual Block 1: 32 → 64 channels + MaxPool (8x8)
    ↓
Residual Block 2: 64 → 128 channels + MaxPool (2x2)
    ↓
Residual Block 3: 128 → 256 channels (1x1)
    ↓
Global Average Pooling: 256 features
    ↓
Fully Connected: 256 → 512 → 256 → 28 classes
    ↓
Output: 28 class probabilities
```

### Key Components

1. **Residual Blocks**: Enable training of deeper networks by allowing gradients to flow directly through shortcut connections
2. **Batch Normalization**: Normalizes activations for stable training
3. **Global Average Pooling**: Reduces spatial dimensions while preserving features
4. **Dropout**: Prevents overfitting by randomly disabling neurons during training

---

## ✨ Key Features

### 1. Proper Data Augmentation
- **Random Rotation**: ±15 degrees (Arabic characters can be slightly rotated)
- **Random Affine**: Translation and scaling variations
- **No Flips**: Avoids horizontal/vertical flips (inappropriate for directional text)
- **Normalization**: Consistent normalization across train/test sets

### 2. Validation Strategy
- **80/20 Split**: 80% training, 20% validation
- **Separate Transforms**: Training uses augmentation, validation doesn't
- **Early Stopping**: Stops if validation accuracy doesn't improve for 10 epochs

### 3. Training Optimizations
- **AdamW Optimizer**: Adam with weight decay for better regularization
- **Learning Rate Scheduling**: Reduces LR when validation plateaus
- **Gradient Clipping**: Prevents exploding gradients
- **Model Checkpointing**: Saves best model automatically

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 2GB+ free disk space

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision pandas matplotlib numpy Pillow
```

### Step 2: Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📖 Usage Guide

### Training the Model

Run the training script:

```bash
python3 train.py
```

**What happens during training:**

1. **Data Loading**: Loads 13,440 training images and splits into train/validation
2. **Model Initialization**: Creates CNN with ~1.5M parameters
3. **Training Loop**: 
   - Trains for up to 100 epochs (stops early if no improvement)
   - Shows real-time metrics: loss, accuracy, learning rate
   - Saves best model when validation accuracy improves
4. **Evaluation**: Tests on 3,360 test images
5. **Error Analysis**: Visualizes misclassified samples

**Expected Output:**
```
Training on: cuda
Starting training...
Train samples: 10752, Val samples: 2688
Epoch [1/100]
  Train Loss: 2.1234, Train Acc: 35.67%
  Val Loss: 1.9876, Val Acc: 42.34%
  LR: 0.000500
  ✓ Best model saved! (Val Acc: 42.34%)
...
```

### Making Predictions

1. **Prepare your image:**
   - Save your Arabic character image as `test.png`
   - Image should be grayscale, any size (will be resized to 32x32)

2. **Run prediction:**
```bash
python3 predict.py
```

**Output:**
```
Model says: ب (confidence: 87.45%)
```

---

## 📚 Understanding the Code

### 1. `dataset.py` - Data Loading

**Purpose**: Loads images and labels from CSV files and applies transformations.

**Key Concepts:**
- **PyTorch Dataset**: Custom class that implements `__len__` and `__getitem__`
- **Transforms**: Preprocessing pipeline (augmentation, normalization)
- **Label Conversion**: Converts 1-indexed labels (1-28) to 0-indexed (0-27)

**How it works:**
```python
# CSV contains flattened 32x32 = 1024 pixel values per row
# Dataset reshapes to 32x32, converts to PIL Image, applies transforms
img = tensor.view(1, 32, 32)  # Reshape
img = ToPILImage()(img)        # Convert for transforms
img = transform(img)           # Apply augmentation/normalization
```

### 2. `model.py` - Neural Network Architecture

**Purpose**: Defines the CNN architecture with residual connections.

**Key Components:**

#### ConvBlock (Residual Block)
```python
# Residual connection allows identity mapping
output = ReLU(conv2(ReLU(conv1(x))) + shortcut(x))
```

**Why Residual Blocks?**
- Enable training of deeper networks
- Allow gradients to flow directly through shortcut
- Help prevent vanishing gradient problem

#### StrongCNN Architecture
- **Progressive Downsampling**: 32x32 → 16x16 → 8x8 → 4x4 → 2x2 → 1x1
- **Progressive Channel Increase**: 1 → 32 → 64 → 128 → 256
- **Global Average Pooling**: Reduces spatial dimensions to 1x1
- **Multi-layer Classifier**: 256 → 512 → 256 → 28 with dropout

### 3. `train.py` - Training Pipeline

**Purpose**: Complete training pipeline with validation, early stopping, and evaluation.

**Training Flow:**

1. **Data Preparation**
   ```python
   # Split data: 80% train, 20% validation
   train_dataset, val_dataset = random_split(...)
   ```

2. **Model Setup**
   ```python
   model = StrongCNN()
   optimizer = AdamW(lr=0.0005, weight_decay=1e-4)
   scheduler = ReduceLROnPlateau(...)
   ```

3. **Training Loop**
   ```python
   for epoch in range(epochs):
       # Training phase
       for batch in train_loader:
           loss = criterion(model(images), labels)
           loss.backward()
           clip_grad_norm_(model.parameters(), 1.0)
           optimizer.step()
       
       # Validation phase
       val_acc = evaluate_model(model, val_loader)
       
       # Save best model
       if val_acc > best_val_acc:
           save_model()
       
       # Early stopping
       if patience_counter >= patience:
           break
   ```

4. **Evaluation**
   ```python
   # Load best model and test
   model.load_state_dict(torch.load("arabic_cnn_gpu.pth"))
   test_acc = evaluate_model(model, test_loader)
   ```

### 4. `predict.py` - Inference

**Purpose**: Load trained model and predict character from single image.

**Process:**
1. Load model weights
2. Preprocess image (resize, normalize)
3. Run inference (forward pass)
4. Get class with highest probability
5. Map class index to Arabic character

---

## 🎓 Training Process Explained

### Phase 1: Data Preparation

**Why 80/20 split?**
- Training set: Used to learn patterns
- Validation set: Used to monitor overfitting and select best model
- Test set: Final evaluation (never used during training)

**Why separate transforms?**
- Training: Needs augmentation to increase dataset diversity
- Validation/Test: Should evaluate on real data (no augmentation)

### Phase 2: Training Loop

**Forward Pass:**
```
Input Image → CNN → Output Logits → Loss Calculation
```

**Backward Pass:**
```
Loss → Gradient Computation → Parameter Update
```

**Key Techniques:**

1. **Gradient Clipping**: Prevents gradients from becoming too large
   ```python
   clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Learning Rate Scheduling**: Reduces LR when model stops improving
   ```python
   scheduler.step(val_accuracy)  # Reduce LR if accuracy plateaus
   ```

3. **Early Stopping**: Prevents overfitting by stopping when validation doesn't improve
   ```python
   if patience_counter >= 10:  # No improvement for 10 epochs
       break
   ```

### Phase 3: Model Selection

**Best Model Selection:**
- Model with highest validation accuracy is saved
- This prevents overfitting to training data
- Test set is only evaluated once at the end

---

## 🧠 Model Architecture Explained

### Why Residual Connections?

Traditional deep networks suffer from the **vanishing gradient problem**: gradients become very small as they propagate backward through many layers, making it hard to train deep networks.

**Residual blocks solve this** by providing a shortcut path:
```
output = F(x) + x  # Shortcut connection
```

If the network needs to learn an identity mapping, it can simply set F(x) = 0, making training easier.

### Architecture Design Choices

1. **Progressive Downsampling**: 
   - Start with full resolution (32x32) to capture fine details
   - Gradually reduce spatial size while increasing channels
   - This captures both local and global features

2. **Global Average Pooling**:
   - Instead of flattening 256x1x1 = 256 values
   - Average pools to 256 values (one per channel)
   - Reduces parameters and helps prevent overfitting

3. **Multi-layer Classifier**:
   - 256 → 512: Expand feature representation
   - 512 → 256: Compress before final classification
   - 256 → 28: Final classification layer
   - Dropout between layers prevents overfitting

### Parameter Count

Total parameters: ~1.5 million
- Convolutional layers: ~1.2M parameters
- Fully connected layers: ~300K parameters

---

## 📊 Expected Results

### Training Metrics

With proper training, you should see:

- **Training Accuracy**: 85-95% (model learns patterns)
- **Validation Accuracy**: 80-90% (generalization to unseen data)
- **Test Accuracy**: 75-90% (final performance on test set)

### Training Time

- **GPU (CUDA)**: ~30-60 minutes for 100 epochs
- **CPU**: ~3-6 hours for 100 epochs

### Model Performance

The model should achieve:
- High accuracy on clear, well-written characters
- Good generalization to variations in handwriting
- Robust to slight rotations and translations

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in train.py
batch_size=64  # Instead of 128
```

**2. Low Accuracy (< 50%)**
- Check data loading: verify CSV files are correct
- Check normalization: ensure transforms are applied correctly
- Increase training epochs or adjust learning rate

**3. Overfitting (High train acc, low val acc)**
- Increase dropout rates in model
- Add more data augmentation
- Reduce model capacity

**4. Underfitting (Low train and val acc)**
- Train for more epochs
- Increase model capacity
- Reduce regularization (dropout, weight decay)

### Debugging Tips

1. **Check data loading:**
   ```python
   dataset = ArabicDataset(...)
   img, label = dataset[0]
   print(f"Image shape: {img.shape}, Label: {label}")
   ```

2. **Verify model output:**
   ```python
   model = StrongCNN()
   x = torch.randn(1, 1, 32, 32)
   output = model(x)
   print(f"Output shape: {output.shape}")  # Should be (1, 28)
   ```

3. **Monitor training:**
   - Watch for decreasing loss
   - Check validation accuracy trends
   - Ensure learning rate is appropriate

---

## 📝 Key Concepts for Students

### 1. Data Augmentation
**Why?** Increases dataset size and helps model generalize to variations.

**What?** Random transformations (rotation, translation, scaling) applied during training.

**Important:** Don't augment validation/test data - evaluate on real data.

### 2. Train/Validation/Test Split
- **Train**: Model learns from this
- **Validation**: Used to tune hyperparameters and select best model
- **Test**: Final evaluation (only used once)

### 3. Early Stopping
**Why?** Prevents overfitting by stopping when model stops improving on validation set.

**How?** Track validation accuracy, stop if it doesn't improve for N epochs.

### 4. Learning Rate Scheduling
**Why?** Large LR at start for fast learning, smaller LR later for fine-tuning.

**How?** Reduce LR when validation accuracy plateaus.

### 5. Gradient Clipping
**Why?** Prevents exploding gradients that can destabilize training.

**How?** Clip gradients to maximum norm (e.g., 1.0).

### 6. Residual Connections
**Why?** Enable training of deeper networks by providing shortcut paths for gradients.

**How?** Add input to output: `output = F(x) + x`

---

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385) - Original residual network paper

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different architectures
- Try different hyperparameters
- Add new features (visualization, metrics, etc.)

---

## 📄 License

This project is for educational purposes.

---

## 👥 Authors

Created for Arabic handwritten character recognition research and education.

---

**Happy Learning! 🚀**
