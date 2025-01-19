import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils import spectral_norm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_spectral_norm=False, norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU(0.2, True)):
        super(ConvBlock, self).__init__()
        layers = []
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        layers.append(conv)
        if activation:
            layers.append(activation)
        if norm_layer:
            layers.append(norm_layer(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_spectral_norm=False, norm_layer=nn.InstanceNorm2d, activation=nn.ReLU(True)):
        super(DeconvBlock, self).__init__()
        layers = []
        deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        if use_spectral_norm:
            deconv = spectral_norm(deconv)
        layers.append(deconv)
        if activation:
            layers.append(activation)
        if norm_layer:
            layers.append(norm_layer(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class LBPGenerator(nn.Module):
    def __init__(self, ngf=64, use_spectral_norm=False):
        super(LBPGenerator, self).__init__()

        # Downsampling layers
        self.down_blocks = nn.ModuleList([
            ConvBlock(2, ngf, use_spectral_norm=use_spectral_norm),
            ConvBlock(ngf, ngf * 2, use_spectral_norm=use_spectral_norm),
            ConvBlock(ngf * 2, ngf * 4, use_spectral_norm=use_spectral_norm),
            ConvBlock(ngf * 4, ngf * 8, use_spectral_norm=use_spectral_norm),
            ConvBlock(ngf * 8, ngf * 8, use_spectral_norm=use_spectral_norm),
            ConvBlock(ngf * 8, ngf * 8, use_spectral_norm=use_spectral_norm),
            ConvBlock(ngf * 8, ngf * 8, use_spectral_norm=use_spectral_norm),
        ])


        layers = [
            nn.LeakyReLU(0.2, True)
        ]

        # Conditionally add spectral normalization
        if use_spectral_norm:
            layers.append(spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=2)))
            layers.append(nn.ReLU(True))
            layers.append(spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=2)))
            layers.append(nn.InstanceNorm2d(ngf * 8))

        else:
            layers.append(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=2))
            layers.append(nn.ReLU(True))
            layers.append(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=2))
            layers.append(nn.InstanceNorm2d(ngf * 8))


        self.bottleneck = nn.Sequential(*layers)
        # Upsampling layers
        self.up_blocks = nn.ModuleList([
            DeconvBlock(ngf * 8 * 2, ngf * 8, use_spectral_norm=use_spectral_norm),
            DeconvBlock(ngf * 8 * 2, ngf * 8, use_spectral_norm=use_spectral_norm),
            DeconvBlock(ngf * 8 * 2, ngf * 8, use_spectral_norm=use_spectral_norm),
            DeconvBlock(ngf * 8 * 2, ngf * 4, use_spectral_norm=use_spectral_norm),
            DeconvBlock(ngf * 4 * 2, ngf * 2, use_spectral_norm=use_spectral_norm),
            DeconvBlock(ngf * 2 * 2, ngf, use_spectral_norm=use_spectral_norm),
            DeconvBlock(ngf * 1 * 2, 1, use_spectral_norm=use_spectral_norm, norm_layer=None, activation=nn.Tanh()),
        ])

    def forward(self, x, mask):
        skip_connections = []
        end = x.clone()

        x = torch.cat([x, 1 - mask], 1)
                      
        # Downsampling
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        for i, up_block in enumerate(self.up_blocks):
            if i < len(skip_connections):
                skip = skip_connections[-(i + 1)]
                x = torch.cat((x, skip), dim=1)
            
            x = up_block(x)

        x = x + end
        x = x * mask + end * (1 - mask)
        return x