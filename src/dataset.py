"""
dataset.py
PyTorch Dataset class for diabetic retinopathy images
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

# Import your preprocessing functions
from preprocessing import preprocess_image


class DRDataset(Dataset):
    """
    Diabetic Retinopathy Dataset
    
    Args:
        image_dir (str): Directory with images
        csv_file (str): Path to CSV with labels (optional)
        transform (callable): Optional transform to apply
        target_size (int): Size to resize images to
    """
    
    def __init__(self, image_dir, csv_file=None, transform=None, target_size=512):
        """
        Initialize dataset
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(list(self.image_dir.rglob(ext)))
        
        self.image_paths = sorted(self.image_paths)
        
        # Load labels if CSV provided
        self.labels_df = None
        if csv_file:
            self.labels_df = pd.read_csv(csv_file, usecols=['Image name', 'Retinopathy grade'])
            self.labels_df = self.labels_df.set_index('Image name')
            print(f"Loaded labels from {csv_file}")
        else:
            print(f"No labels file provided - using dummy labels")
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        """
        Returns total number of images
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and return one image and its label
        
        Args:
            idx (int): Index of image to load
            
        Returns:
            tuple: (image, label)
        """
        # Get image path
        img_path = self.image_paths[idx]
        
        # Load and preprocess image
        image = preprocess_image(img_path, self.target_size)
        
        # Convert to torch tensor (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Apply additional transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Get label
        if self.labels_df is not None:
            img_name = img_path.stem  # e.g. "IDRiD_001"
            label = int(self.labels_df.loc[img_name, 'Retinopathy grade'])
        else:
            label = 0  # dummy
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label


def test_dataset():
    """
    Test function to verify dataset works
    """
    from pathlib import Path
    
    # Create dataset
    data_dir = Path('../data/raw/idrid')
    dataset = DRDataset(
        image_dir=data_dir,
        csv_file=None,  # No labels for now
        target_size=512
    )
    
    print(f"\n{'='*60}")
    print("DATASET TEST")
    print(f"{'='*60}")
    
    # Test basic properties
    print(f"Dataset size: {len(dataset)} images")
    
    # Load first image
    print(f"\nLoading first image...")
    image, label = dataset[0]
    
    print(f"Image shape: {image.shape}")  # Should be (3, 512, 512)
    print(f"Image dtype: {image.dtype}")  # Should be float32
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")  # Should be [0, 1]
    print(f"Label: {label}")
    
    # Load a few more
    print(f"\nLoading 5 random images...")
    import random
    for i in random.sample(range(len(dataset)), 5):
        img, lbl = dataset[i]
        print(f"  Image {i}: shape={img.shape}, label={lbl}")
    
    print(f"\n{'='*60}")
    print("✅ DATASET TEST PASSED!")
    print(f"{'='*60}")
    
    return dataset


if __name__ == "__main__":
    # Test when running this file directly
    test_dataset()