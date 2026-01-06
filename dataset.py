"""
Arabic Handwritten Character Dataset Loader

This module provides a PyTorch Dataset class for loading Arabic handwritten character
images from CSV files. The dataset handles image preprocessing and transformation
for training and evaluation.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms


class ArabicDataset(Dataset):
    """
    PyTorch Dataset for Arabic handwritten character recognition.
    
    Loads image and label data from CSV files, applies transformations,
    and provides a standard PyTorch Dataset interface for training.
    
    Args:
        img_csv (str): Path to CSV file containing image pixel values (flattened 32x32 images)
        label_csv (str): Path to CSV file containing corresponding labels
        transform (callable, optional): Transform to apply to images (augmentation, normalization, etc.)
    
    Attributes:
        images (numpy.ndarray): Array of image pixel values
        labels (numpy.ndarray): Array of character labels
        transform (callable): Image transformation pipeline
    """
    
    def __init__(self, img_csv, label_csv, transform=None):
        """
        Initialize the dataset by loading images and labels from CSV files.
        
        Args:
            img_csv (str): Path to CSV file with image data (each row is a flattened 32x32 image)
            label_csv (str): Path to CSV file with label data (one label per row)
            transform (callable, optional): Image transformation pipeline
        """
        self.images = pd.read_csv(img_csv, header=None).values
        self.labels = pd.read_csv(label_csv, header=None).values
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a single sample (image, label) from the dataset.
        
        The image is loaded from CSV, reshaped to 32x32, converted to PIL Image,
        and then transformed. Labels are converted from 1-indexed to 0-indexed.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, label) where:
                - image_tensor: Preprocessed image as a PyTorch tensor
                - label: Integer label (0-27) representing the Arabic character class
        """
        # Load image data and reshape to 32x32 grayscale format
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        img = img.view(1, 32, 32)  # Reshape flattened array to 32x32 image (ToTensor will normalize)

        # Convert tensor to PIL Image for transformation pipeline compatibility
        to_pil = transforms.ToPILImage()
        img = to_pil(img)

        # Apply transformations (augmentation, normalization, etc.) if provided
        if self.transform:
            img = self.transform(img)

        # Convert label from 1-indexed to 0-indexed (CSV uses 1-28, PyTorch uses 0-27)
        label = int(self.labels[idx]) - 1
        return img, label
