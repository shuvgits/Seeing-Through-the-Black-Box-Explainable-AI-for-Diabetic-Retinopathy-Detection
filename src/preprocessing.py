"""
preprocessing.py
Image preprocessing functions for diabetic retinopathy detection
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def load_image(image_path):
    """
    Load an image from file path.
    
    Args:
        image_path (str or Path): Path to image file
        
    Returns:
        numpy.ndarray: Image as numpy array (H, W, C) in RGB format
    """
    # Load with PIL (handles different formats well)
    img = Image.open(image_path)
    
    # Convert to RGB (in case image is grayscale or has alpha channel)
    img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    return img_array


def resize_image(image, target_size=512):
    """
    Resize image to target_size × target_size.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (int): Target dimension for both width and height
        
    Returns:
        numpy.ndarray: Resized image
    """
    # Get current dimensions
    h, w = image.shape[:2]
    
    # Resize using OpenCV (high quality interpolation)
    # cv2.INTER_AREA is best for shrinking images
    resized = cv2.resize(
        image, 
        (target_size, target_size), 
        interpolation=cv2.INTER_AREA
    )
    
    return resized


def normalize_image(image):
    """
    Normalize pixel values from [0, 255] to [0, 1].
    
    Args:
        image (numpy.ndarray): Input image with values in [0, 255]
        
    Returns:
        numpy.ndarray: Normalized image with values in [0, 1]
    """
    # Convert to float and divide by 255
    normalized = image.astype(np.float32) / 255.0
    
    return normalized


def crop_black_borders(image, threshold=10):
    """
    Crop black borders around fundus images (optional).
    Many fundus images have black borders around the circular retina.
    
    Args:
        image (numpy.ndarray): Input image
        threshold (int): Pixel intensity threshold for detecting content
        
    Returns:
        numpy.ndarray: Cropped image
    """
    # Convert to grayscale for border detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Find rows and columns that have content (not black)
    rows = np.any(gray > threshold, axis=1)
    cols = np.any(gray > threshold, axis=0)
    
    # Get bounding box
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Crop image
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped


def preprocess_image(image_path, target_size=512, crop_borders=False):
    """
    Complete preprocessing pipeline.
    
    Args:
        image_path (str or Path): Path to image
        target_size (int): Target size for resizing
        crop_borders (bool): Whether to crop black borders
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model
    """
    # Step 1: Load image
    img = load_image(image_path)
    
    # Step 2: Crop borders (optional)
    if crop_borders:
        img = crop_black_borders(img)
    
    # Step 3: Resize
    img = resize_image(img, target_size)
    
    # Step 4: Normalize
    img = normalize_image(img)
    
    return img


def get_augmentation_transforms():
    """
    Get PyTorch augmentation transforms for training.
    
    Returns:
        torchvision.transforms.Compose: Augmentation pipeline
    """
    transform = transforms.Compose([
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Random vertical flip
        transforms.RandomVerticalFlip(p=0.5),
        
        # Random rotation (±15 degrees)
        transforms.RandomRotation(degrees=15),
        
        # Random brightness and contrast
        transforms.ColorJitter(
            brightness=0.2,  # ±20% brightness
            contrast=0.2,    # ±20% contrast
        ),
        
        # Convert to tensor
        transforms.ToTensor(),
    ])
    
    return transform


def visualize_preprocessing(image_path, target_size=512):
    """
    Visualize before and after preprocessing.
    
    Args:
        image_path (str or Path): Path to image
        target_size (int): Target size for preprocessing
    """
    # Load original
    original = load_image(image_path)
    
    # Preprocess
    preprocessed = preprocess_image(image_path, target_size)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title(f'Original\nSize: {original.shape[1]}×{original.shape[0]}')
    axes[0].axis('off')
    
    # Preprocessed image
    axes[1].imshow(preprocessed)
    axes[1].set_title(f'Preprocessed\nSize: {target_size}×{target_size}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Original shape: {original.shape}")
    print(f"Original value range: [{original.min()}, {original.max()}]")
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Preprocessed value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")


# Test the preprocessing if running this file directly
if __name__ == "__main__":
    from pathlib import Path
    
    # Find a sample image
    data_path = Path('../data/raw/idrid')
    sample_images = list(data_path.rglob('*.jpg'))
    
    if len(sample_images) > 0:
        print("Testing preprocessing on sample image...")
        visualize_preprocessing(sample_images[0])
    else:
        print("No images found for testing")