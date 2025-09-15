"""
Script to test the generator and discriminator models.
"""
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from src.config import config
from src.modeling import create_models


def test_models():
    """Test the generator and discriminator models"""
    print("Testing models...")
    print(f"Using device: {config.DEVICE}")

    # Create models
    generator, discriminator = create_models(config.DEVICE)

    # Print model architectures
    print("\nGenerator architecture:")
    print(generator)
    print(f"\nTotal generator parameters: {sum(p.numel() for p in generator.parameters()):,}")

    print("\nDiscriminator architecture:")
    print(discriminator)
    print(f"\nTotal discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Test with sample input
    batch_size = 2
    sample_L = torch.randn(batch_size, 1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]).to(config.DEVICE)
    sample_ab = torch.randn(batch_size, 2, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]).to(config.DEVICE)

    # Test generator
    with torch.no_grad():
        generated_ab = generator(sample_L)
        print(f"\nGenerator input shape: {sample_L.shape}")
        print(f"Generator output shape: {generated_ab.shape}")

        # Test discriminator
        real_output = discriminator(sample_L, sample_ab)
        fake_output = discriminator(sample_L, generated_ab)
        print(f"Discriminator real output shape: {real_output.shape}")
        print(f"Discriminator fake output shape: {fake_output.shape}")

        # Check if outputs are reasonable
        print(f"Generated values range: [{generated_ab.min():.3f}, {generated_ab.max():.3f}]")
        print(f"Real output range: [{real_output.min():.3f}, {real_output.max():.3f}]")
        print(f"Fake output range: [{fake_output.min():.3f}, {fake_output.max():.3f}]")

    print("\nModel test completed successfully!")


if __name__ == "__main__":
    test_models()