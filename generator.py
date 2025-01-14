import torch.nn as nn

class ImageGenerator(nn.Module):
    def __init__(self, nfg, use_spectral_norm, device):
        super(ImageGenerator, self).__init__()
        
        self.ngf = nfg
        self.use_spectral_norm = use_spectral_norm

        # Define downsampling layers
        self.down_blocks = nn.ModuleList([
            self._create_down_block(in_channels, out_channels)
            for in_channels, out_channels in zip([5, self.ngf, self.ngf * 2,  self.ngf * 4, self.ngf * 8, self.ngf * 8, self.ngf * 8],
                                                 [self.ngf, self.ngf * 2,  self.ngf * 4,  self.ngf * 8, self.ngf * 8, self.ngf * 8, self.ngf * 8])
        ])

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(self.ngf * 8, self.ngf * 8, kernel_size=4, stride=2, padding=1), self.use_spectral_norm),
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(self.ngf * 8, self.ngf * 8, kernel_size=4, stride=2, padding=1), self.use_spectral_norm),
            nn.InstanceNorm2d(self.ngf * 8),
        )

        # Define upsampling layers
        self.up_blocks = nn.ModuleList([
            self._create_up_block(in_channels, out_channels)
            for in_channels, out_channels in zip([self.ngf * 8 * 2, self.ngf * 8 * 2, self.ngf * 8 * 2, self.ngf * 8 * 2, self.ngf * 4 * 3, self.ngf * 4 , self.ngf *2],
                                                 [self.ngf * 8, self.ngf * 8, self.ngf * 8, self.ngf * 4, self.nfg * 2 , self.ngf, 3])
        ])

        # Custom attention mechanism
        self.attention = MyAttention(opt)

    def _create_down_block(self, in_channels, out_channels):
        """Creates a downsampling block."""
        return nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), self.use_spectral_norm),
            nn.InstanceNorm2d(out_channels)
        )

    def _create_up_block(self, in_channels, out_channels):
        """Creates an upsampling block."""
        layers = [
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), self.use_spectral_norm),
        ]
        if out_channels != 3:
            layers.append(nn.InstanceNorm2d(out_channels))
        else:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x, lbp, mask):
        """Forward pass of the ImageGenerator."""
        # Downsampling

        inputs = torch.cat([x, lbp, 1 - mask], dim=1)
        features = []
        for down_block in self.down_blocks:
            inputs = down_block(inputs)
            features.append(inputs)

        # Bottleneck
        bottleneck_output = self.bottleneck(features[-1])

        # Upsampling
        up_input = bottleneck_output
        list_index = 0

        for idx, up_block in enumerate(self.up_blocks[:-1]):

            if idx == 4:
                skip_connection = features[-(idx + 1)]
                up_input = torch.cat([up_input, skip_connection], dim=1)
                up_input = nn.ReLU(inplace=True)(up_input)
                
                attention_input = torch.cat([up_input[:, :256], up_input[:, 256:]], dim=1)
                up_input = self.attention(attention_input, attention_input, mask)

                continue

            elif idx == 5:
                up_input = self.up_block(up_input)
                continue



            skip_connection = features[-(list_index + 1)]
            up_input = torch.cat([up_input, skip_connection], dim=1)
            up_input = up_block(up_input)
            list_index +=1

        final_output = up_input + x

        # Mask the output
        return final_output * mask + x * (1 - mask)
