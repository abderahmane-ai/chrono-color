import os
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import argparse

from config import config
from data_loading.transforms import RGBToLAB, LABToTensor, Resize


def create_processed_dataset():
    """
    Preprocess all images in the raw data directory and save them as tensors
    in the processed directory, maintaining the same folder structure.
    """
    # Create processed directory structure
    for decade in config.DECADES:
        decade_processed_dir = config.PROCESSED_DATA_DIR / decade
        decade_processed_dir.mkdir(parents=True, exist_ok=True)

    # Initialize transforms
    resize_transform = Resize(config.IMAGE_SIZE)
    rgb_to_lab = RGBToLAB()
    lab_to_tensor = LABToTensor()

    # Process each decade folder
    for decade in config.DECADES:
        decade_raw_dir = config.RAW_DATA_DIR / decade
        decade_processed_dir = config.PROCESSED_DATA_DIR / decade

        if not decade_raw_dir.exists():
            print(f"Warning: Raw directory {decade_raw_dir} does not exist. Skipping.")
            continue

        # Get all image files in the decade folder
        image_files = [f for f in os.listdir(decade_raw_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Processing {len(image_files)} images from {decade}...")

        # Process each image
        for img_file in tqdm(image_files, desc=decade):
            img_path = decade_raw_dir / img_file

            try:
                # Load and process the image
                img = Image.open(img_path).convert('RGB')
                img = resize_transform(img)
                lab_img = rgb_to_lab(img)
                lab_tensor = lab_to_tensor(lab_img)

                # Split into L and ab channels
                L_channel = lab_tensor[0:1, :, :]  # Shape: (1, H, W)
                ab_channels = lab_tensor[1:, :, :]  # Shape: (2, H, W)

                # Save as tensors
                base_name = os.path.splitext(img_file)[0]
                torch.save(L_channel, decade_processed_dir / f"{base_name}_L.pt")
                torch.save(ab_channels, decade_processed_dir / f"{base_name}_ab.pt")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue


def create_processed_dataset_cli():
    """Command line interface for creating the processed dataset."""
    parser = argparse.ArgumentParser(description='Preprocess images for ChronoColor project')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if files exist')
    args = parser.parse_args()

    # Check if processed data already exists
    if not args.force and any(config.PROCESSED_DATA_DIR.iterdir()):
        response = input("Processed data already exists. Reprocess? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return

    print("Starting image preprocessing...")
    create_processed_dataset()
    print("Preprocessing complete!")


if __name__ == "__main__":
    create_processed_dataset_cli()