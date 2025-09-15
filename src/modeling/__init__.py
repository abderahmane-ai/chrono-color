from .generator import Generator, init_weights_normal
from .discriminator import Discriminator


def create_models(device):
    """Create and initialize generator and discriminator models"""
    generator = Generator()
    discriminator = Discriminator()

    # Initialize weights
    generator.apply(init_weights_normal)
    discriminator.apply(init_weights_normal)

    # Move to device
    generator.to(device)
    discriminator.to(device)

    return generator, discriminator