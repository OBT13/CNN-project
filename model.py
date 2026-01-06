"""
Enhanced CNN Model for Arabic Handwritten Character Recognition

This module implements a deep convolutional neural network with residual connections
for recognizing 28 Arabic handwritten characters. The architecture uses residual blocks
to enable deeper networks and better gradient flow during training.
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional block with residual connections.
    
    Implements a residual block pattern: two convolutional layers with batch normalization
    and ReLU activation, plus a shortcut connection that allows gradients to flow
    directly through the network. This helps train deeper networks effectively.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for the first convolution (default: 1)
    
    Attributes:
        conv1, conv2: Convolutional layers
        bn1, bn2: Batch normalization layers
        shortcut: Residual connection (1x1 conv if dimensions change)
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the convolutional block with residual connection.
        
        Args:
            in_channels (int): Number of input feature channels
            out_channels (int): Number of output feature channels
            stride (int): Stride for downsampling (1 = no downsampling, 2 = halve spatial size)
        """
        super().__init__()
        # First convolutional layer (may downsample with stride > 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer (same size, no downsampling)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection: identity if dimensions match, otherwise 1x1 conv to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Need to adjust dimensions for residual connection
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Computes: output = ReLU(conv2(ReLU(conv1(x))) + shortcut(x))
        The residual connection allows the network to learn identity mappings,
        making it easier to train deeper networks.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, H', W')
        """
        # Save input for residual connection
        residual = self.shortcut(x)
        
        # Main path: two convolutions with batch norm and ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        return out


class StrongCNN(nn.Module):
    """
    Enhanced CNN architecture for Arabic handwritten character recognition.
    
    This network uses a deep architecture with residual blocks to extract features
    from 32x32 grayscale images and classify them into 28 Arabic character classes.
    The architecture progressively reduces spatial dimensions while increasing
    feature channels, then uses global average pooling and fully connected layers
    for classification.
    
    Architecture:
        - Initial conv: 1 channel -> 32 channels (32x32)
        - Block 1: 32 -> 64 channels with downsampling (16x16 -> 8x8)
        - Block 2: 64 -> 128 channels with downsampling (8x8 -> 4x4 -> 2x2)
        - Block 3: 128 -> 256 channels with downsampling (2x2 -> 1x1)
        - Global Average Pooling: 256x1x1 -> 256
        - Classifier: 256 -> 512 -> 256 -> 28 classes
    
    Args:
        num_classes (int): Number of output classes (default: 28 for Arabic letters)
    """
    
    def __init__(self, num_classes=28):
        """
        Initialize the CNN model architecture.
        
        Args:
            num_classes (int): Number of character classes to classify (default: 28)
        """
        super().__init__()

        # Initial convolutional layer: converts grayscale input to feature maps
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1 channel -> 32 channels, maintains 32x32 size
            nn.BatchNorm2d(32),  # Normalize activations for stable training
            nn.ReLU(inplace=True)  # Non-linearity activation
        )

        # Block 1: First residual block with downsampling
        # Input: 32 channels @ 32x32 -> Output: 64 channels @ 8x8
        self.block1 = ConvBlock(32, 64, stride=2)  # stride=2 halves spatial dimensions
        self.pool1 = nn.MaxPool2d(2)  # Additional downsampling: 16x16 -> 8x8

        # Block 2: Second residual block with downsampling
        # Input: 64 channels @ 8x8 -> Output: 128 channels @ 2x2
        self.block2 = ConvBlock(64, 128, stride=2)  # 8x8 -> 4x4
        self.pool2 = nn.MaxPool2d(2)  # 4x4 -> 2x2

        # Block 3: Third residual block with downsampling
        # Input: 128 channels @ 2x2 -> Output: 256 channels @ 1x1
        self.block3 = ConvBlock(128, 256, stride=2)  # 2x2 -> 1x1

        # Global Average Pooling: reduces spatial dimensions to 1x1 while preserving channels
        # This helps prevent overfitting and reduces parameters compared to flattening
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected classifier: maps features to class probabilities
        # Multiple layers with dropout for regularization
        self.fc = nn.Sequential(
            nn.Dropout(0.4),  # Dropout to prevent overfitting
            nn.Linear(256, 512),  # Expand feature representation
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Higher dropout in middle layer
            nn.Linear(512, 256),  # Compress before final classification
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Lower dropout before output
            nn.Linear(256, num_classes)  # Final classification layer
        )

    def forward(self, x):
        """
        Forward pass through the entire network.
        
        Processes input images through convolutional feature extraction,
        global pooling, and classification layers.
        
        Args:
            x (torch.Tensor): Input batch of images, shape (batch_size, 1, 32, 32)
            
        Returns:
            torch.Tensor: Class logits, shape (batch_size, num_classes)
        """
        # Feature extraction through residual blocks
        x = self.initial_conv(x)  # 32x32x32
        x = self.block1(x)  # 16x16x64
        x = self.pool1(x)  # 8x8x64
        x = self.block2(x)  # 4x4x128
        x = self.pool2(x)  # 2x2x128
        x = self.block3(x)  # 1x1x256
        
        # Global average pooling: 1x1x256 -> 256
        x = self.global_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        return x
