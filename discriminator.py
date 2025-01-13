import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import functools


def get_norm_layer(norm_type='instance'):
    """
    Get the normalization layer by type.
    """
    norm_layers = {
        'batch': functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True),
        'instance': functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
        'none': None
    }
    if norm_type not in norm_layers:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

    return norm_layers[norm_type]


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_spectral_norm=True):
        """
        Define a multi-layer PatchGAN discriminator.
        """
        super().__init__()
        layers = []
        kw, padw = 4, 1
        use_bias = isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm2d

        # First layer
        layers.append(self.add_spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), use_spectral_norm))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                self.add_spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    use_spectral_norm
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            self.add_spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                use_spectral_norm
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            self.add_spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), use_spectral_norm)
        ]

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    @staticmethod
    def add_spectral_norm(layer, use_spectral_norm):
        """
        Apply spectral normalization if required.
        """
        return spectral_norm(layer) if use_spectral_norm else layer

    def forward(self, x):
        return self.model(x)


def define_discriminator(input_nc, ndf, device='cuda', norm_type='instance', init_type='normal', gain=0.02):
    """
    Create and initialize a PatchGAN discriminator.
    """
    norm_layer = get_norm_layer(norm_type)
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=False, use_spectral_norm=True)
    return netD 

