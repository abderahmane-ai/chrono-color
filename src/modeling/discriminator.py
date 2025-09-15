import torch
import torch.nn as nn
from config import config


class Discriminator(nn.Module):
    """PatchGAN discriminator for image colorization"""

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, config.DISC_FILTERS, normalization=False),
            *discriminator_block(config.DISC_FILTERS, config.DISC_FILTERS * 2),
            *discriminator_block(config.DISC_FILTERS * 2, config.DISC_FILTERS * 4),
            *discriminator_block(config.DISC_FILTERS * 4, config.DISC_FILTERS * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(config.DISC_FILTERS * 8, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# We can use the same weight initialization function
from .generator import init_weights_normal

