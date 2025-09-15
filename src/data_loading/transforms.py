import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import random


class RGBToLAB:
    """
    Convert a PIL Image or numpy array from RGB to L*a*b* color space.
    """

    def __call__(self, img):
        img_np = np.array(img).astype(np.uint8)
        lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab_img = lab_img.transpose(2, 0, 1).astype(np.float32)
        lab_img[0] = lab_img[0] * (100.0 / 255.0)  # L channel
        lab_img[1:] = lab_img[1:] - 128.0  # a* and b* channels
        return lab_img


class LABToTensor:
    """
    Convert a L*a*b* numpy array to PyTorch tensors and normalize.
    """

    def __call__(self, lab_img):
        lab_tensor = torch.from_numpy(lab_img).float()
        lab_tensor[0] = (lab_tensor[0] / 50.0) - 1.0  # L: [0, 100] -> [-1, 1]
        lab_tensor[1:] = lab_tensor[1:] / 128.0  # ab: [-128, 127] -> [-1, 1]
        return lab_tensor


class Resize:
    """Resize the input image to the given size."""

    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(size)

    def __call__(self, img):
        return self.resize(img)


class RandomCrop:
    """Randomly crop the image to the given size."""

    def __init__(self, size):
        self.size = size
        self.crop = transforms.RandomCrop(size)

    def __call__(self, img):
        return self.crop(img)


class RandomHorizontalFlip:
    """Randomly flip the image horizontally."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img


class ColorJitter:
    """
    Randomly change the brightness, contrast, saturation, and hue of an image.
    This is applied before converting to LAB space.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, img):
        return self.jitter(img)


def get_transforms(mode='train', size=(256, 256)):
    """
    Get appropriate transforms for training, validation, or testing.
    """
    base_transforms = [
        Resize(size),
        RGBToLAB(),
        LABToTensor()
    ]

    if mode == 'train':
        # Add augmentations only for training
        augmentations = [
            RandomCrop(size),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ]
        # Apply augmentations before base transforms
        return transforms.Compose(augmentations + base_transforms)
    else:
        # For validation and testing, just use base transforms
        return transforms.Compose(base_transforms)


def denormalize_lab(lab_tensor):
    """
    Denormalize LAB tensor back to original LAB ranges.
    L: [-1, 1] -> [0, 100]
    ab: [-1, 1] -> [-128, 127]
    """
    lab_denorm = lab_tensor.clone()
    lab_denorm[0] = (lab_tensor[0] + 1.0) * 50.0  # L channel
    lab_denorm[1:] = lab_tensor[1:] * 128.0  # ab channels
    return lab_denorm


def lab_to_rgb(lab_tensor):
    """
    Convert LAB tensor to RGB.
    Input: LAB tensor in normalized range [-1, 1]
    Output: RGB tensor in range [0, 1]
    """
    # Denormalize LAB
    lab_denorm = denormalize_lab(lab_tensor)
    
    # Convert to numpy and rearrange dimensions
    lab_np = lab_denorm.cpu().numpy().transpose(1, 2, 0)
    
    # Convert LAB to RGB
    lab_np = lab_np.astype(np.uint8)
    rgb_np = cv2.cvtColor(lab_np, cv2.COLOR_LAB2RGB)
    
    # Convert back to tensor and normalize to [0, 1]
    rgb_tensor = torch.from_numpy(rgb_np).float() / 255.0
    rgb_tensor = rgb_tensor.permute(2, 0, 1)
    
    return rgb_tensor