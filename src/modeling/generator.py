import torch
import torch.nn as nn
from config import config


class UNetDown(nn.Module):
    """U-Net downsampling block with optional dropout"""

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """U-Net upsampling block"""

    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    """U-Net based generator for image colorization"""

    def __init__(self, in_channels=1, out_channels=2):
        super(Generator, self).__init__()

        # Downsampling (encoder)
        self.down1 = UNetDown(in_channels, config.GEN_FILTERS, normalize=False)
        self.down2 = UNetDown(config.GEN_FILTERS, config.GEN_FILTERS * 2)
        self.down3 = UNetDown(config.GEN_FILTERS * 2, config.GEN_FILTERS * 4)
        self.down4 = UNetDown(config.GEN_FILTERS * 4, config.GEN_FILTERS * 8, dropout=config.DROPOUT_RATE)
        self.down5 = UNetDown(config.GEN_FILTERS * 8, config.GEN_FILTERS * 8, dropout=config.DROPOUT_RATE)
        self.down6 = UNetDown(config.GEN_FILTERS * 8, config.GEN_FILTERS * 8, dropout=config.DROPOUT_RATE)
        self.down7 = UNetDown(config.GEN_FILTERS * 8, config.GEN_FILTERS * 8, dropout=config.DROPOUT_RATE)
        self.down8 = UNetDown(config.GEN_FILTERS * 8, config.GEN_FILTERS * 8, normalize=False,
                              dropout=config.DROPOUT_RATE)

        # Upsampling (decoder)
        self.up1 = UNetUp(config.GEN_FILTERS * 8, config.GEN_FILTERS * 8, dropout=config.DROPOUT_RATE)
        self.up2 = UNetUp(config.GEN_FILTERS * 8 * 2, config.GEN_FILTERS * 8, dropout=config.DROPOUT_RATE)
        self.up3 = UNetUp(config.GEN_FILTERS * 8 * 2, config.GEN_FILTERS * 8, dropout=config.DROPOUT_RATE)
        self.up4 = UNetUp(config.GEN_FILTERS * 8 * 2, config.GEN_FILTERS * 8, dropout=config.DROPOUT_RATE)
        self.up5 = UNetUp(config.GEN_FILTERS * 8 * 2, config.GEN_FILTERS * 4)
        self.up6 = UNetUp(config.GEN_FILTERS * 4 * 2, config.GEN_FILTERS * 2)
        self.up7 = UNetUp(config.GEN_FILTERS * 2 * 2, config.GEN_FILTERS)

        # Final layer
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(config.GEN_FILTERS * 2, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections
        d1 = self.down1(x)  # 128
        d2 = self.down2(d1)  # 64
        d3 = self.down3(d2)  # 32
        d4 = self.down4(d3)  # 16
        d5 = self.down5(d4)  # 8
        d6 = self.down6(d5)  # 4
        d7 = self.down7(d6)  # 2
        d8 = self.down8(d7)  # 1

        u1 = self.up1(d8, d7)  # 2
        u2 = self.up2(u1, d6)  # 4
        u3 = self.up3(u2, d5)  # 8
        u4 = self.up4(u3, d4)  # 16
        u5 = self.up5(u4, d3)  # 32
        u6 = self.up6(u5, d2)  # 64
        u7 = self.up7(u6, d1)  # 128

        return self.final(u7)


def init_weights_normal(m):
    """Initialize weights with normal distribution with mean=0, std=0.02"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)