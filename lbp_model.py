import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils import spectral_norm

self.optimizer_LBP = torch.optim.Adam(self.netLBP.parameters(), lr=opt.lr, betas=(0.5, 0.999))
self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(0.5, 0.999))

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

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(ngf * 8),
        )

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

    def forward(self, x):
        skip_connections = []

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

        return x