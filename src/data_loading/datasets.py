import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

from config import config
from .transforms import RGBToLAB, LABToTensor, Resize


class HistoricalColorDataset(Dataset):
    """
    Custom Dataset for loading historical color images organized by decade.
    """

    def __init__(self, root_dir, transform=None, mode='train', decades=None):
        """
        Args:
            root_dir (string): Directory with decade subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): One of 'train', 'val', 'test'. Determines which augmentations to apply.
            decades (list): List of decades to include. If None, includes all decades.
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.decades = decades if decades is not None else config.DECADES

        # Collect all image paths from the specified decade folders
        self.image_paths = []
        for decade in self.decades:
            decade_path = self.root_dir / decade
            if decade_path.exists():
                image_files = [f for f in os.listdir(decade_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                self.image_paths.extend([decade_path / f for f in image_files])
                print(f"[DEBUG] Found {len(image_files)} images in {decade_path}")
            else:
                print(f"[DEBUG] Decade path does not exist: {decade_path}")

        # Store decade information for each image
        self.image_decades = []
        for path in self.image_paths:
            self.image_decades.append(path.parent.name)

        # Define base transforms
        base_transforms = [
            Resize(config.IMAGE_SIZE),
            RGBToLAB(),
            LABToTensor()
        ]

        # Add training-specific augmentations
        if mode == 'train' and transform is None:
            # We'll add more augmentations later
            self.transform = transforms.Compose(base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the color image
        global color_image
        img_path = self.image_paths[idx]
        decade = self.image_decades[idx]

        try:
            color_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

        # Apply transforms
        lab_tensor = self.transform(color_image)

        # Split the tensor into L (input) and ab (target)
        L_channel = lab_tensor[0:1, :, :]  # Grayscale input
        ab_channels = lab_tensor[1:, :, :]  # Color target

        return L_channel, ab_channels, decade


def get_dataloaders(data_dir=None, batch_size=None, decades=None):
    """Helper function to create training, validation, and test dataloaders."""
    
    # Use defaults if not provided
    if data_dir is None:
        data_dir = config.RAW_DATA_DIR
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Create full dataset
    full_dataset = HistoricalColorDataset(
        root_dir=data_dir,
        mode='all',
        decades=decades
    )

    # Calculate split sizes
    dataset_size = len(full_dataset)
    print(f"[DEBUG] Full dataset size: {dataset_size} images from {data_dir}")
    
    if dataset_size == 0:
        raise ValueError(f"No images found in dataset at {data_dir}. Please check data directory and file extensions.")
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Assign mode to each subset
    train_dataset.mode = 'train'
    val_dataset.mode = 'val'
    test_dataset.mode = 'test'

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_decade_dataloaders(data_dir=None, batch_size=None):
    """
    Create separate dataloaders for each decade.
    Returns a dictionary where keys are decades and values are dataloaders.
    """
    # Use defaults if not provided
    if data_dir is None:
        data_dir = config.RAW_DATA_DIR
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    decade_loaders = {}

    for decade in config.DECADES:
        decade_dataset = HistoricalColorDataset(
            root_dir=data_dir,
            mode='all',
            decades=[decade]
        )

        decade_loader = DataLoader(
            decade_dataset, batch_size=batch_size, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=True
        )

        decade_loaders[decade] = decade_loader

    return decade_loaders