"""
Arabic Character Prediction Script

This script loads a trained CNN model and performs inference on a single image
to predict which Arabic handwritten character it represents. The script outputs
the predicted character and confidence score.
"""

import torch
from model import StrongCNN
from PIL import Image
import torchvision.transforms as transforms

# Mapping from class index to Arabic character
# Classes 0-27 correspond to the 28 Arabic letters
arabic_letters = [
    "ا","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز",
    "س","ش","ص","ض","ط","ظ","ع","غ","ف","ق",
    "ك","ل","م","ن","ه","و","ي"
]


def load_model(model_path="arabic_cnn_gpu.pth", device=None):
    """
    Load a trained CNN model from disk.
    
    Args:
        model_path (str): Path to the saved model weights file
        device (str, optional): Device to load model on ('cuda' or 'cpu').
                               If None, automatically detects available device.
    
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = StrongCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    return model, device


def preprocess_image(image_path, size=(32, 32)):
    """
    Load and preprocess an image for model inference.
    
    The preprocessing pipeline matches the test transform used during training:
    - Convert to grayscale
    - Resize to 32x32
    - Convert to tensor and normalize to [-1, 1]
    
    Args:
        image_path (str): Path to the input image file
        size (tuple): Target image size (width, height), default (32, 32)
    
    Returns:
        torch.Tensor: Preprocessed image tensor, shape (1, 1, 32, 32)
    """
    # Image preprocessing pipeline (matches test_transform from training)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor and scale to [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    # Load image and convert to grayscale
    img = Image.open(image_path).convert("L")
    img = img.resize(size)
    
    # Apply transformations and add batch dimension
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension: (1, 32, 32) -> (1, 1, 32, 32)
    return img_tensor


def predict_character(model, image_tensor, device):
    """
    Predict the Arabic character from a preprocessed image tensor.
    
    Args:
        model (torch.nn.Module): Trained CNN model
        image_tensor (torch.Tensor): Preprocessed image tensor, shape (1, 1, 32, 32)
        device (str): Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        tuple: (predicted_character, confidence_score)
            - predicted_character (str): Arabic letter prediction
            - confidence_score (float): Prediction confidence as percentage (0-100)
    """
    # Move tensor to appropriate device
    image_tensor = image_tensor.to(device)
    
    # Run inference (no gradient computation needed)
    with torch.no_grad():
        output = model(image_tensor)  # Get raw logits
        pred = output.argmax(1).item()  # Get predicted class index
        
        # Calculate confidence as softmax probability
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0][pred].item() * 100
    
    # Map class index to Arabic character
    predicted_character = arabic_letters[pred]
    return predicted_character, confidence


def main():
    """
    Main function to run character prediction on a test image.
    
    Loads the model, preprocesses the input image, and prints the prediction.
    """
    # Load trained model
    model, device = load_model("arabic_cnn_gpu.pth")
    
    # Preprocess input image
    image_tensor = preprocess_image("test.png")
    
    # Get prediction
    character, confidence = predict_character(model, image_tensor, device)
    
    # Display results
    print(f"Model says: {character} (confidence: {confidence:.2f}%)")


if __name__ == "__main__":
    main()
