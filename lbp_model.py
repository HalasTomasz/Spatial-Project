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
    def __init__(self):
        super(LBPGenerator, self).__init__()
        ngf = 64
        use_spectral_norm = False

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



class LBP(pl.LightningModule):

    def __init__(self, model, model_config, discrimantor, gan_loss):
        super(LPD, self).__init__()

        self.model = model
        self.model_config = model_config
        self.discrimantor = discrimantor

        self.muli_level_loss = nn.MultiLevelLoss()
        self.reconstruction_loss = nn.MSELoss()
        self.adversal_loss = gan_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        L_with_hole, mask = batch
        L_filled_gen = self.forward(L_with_hole)
        
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def update_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def freeze_layers(self):
        """
        Set requires_grad to False for all parameters of the model.
        Customize as needed to freeze specific layers.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        print("All layers have been frozen.")
   
    def loss(self, input_I, output_I_o):

        pred_I_o = self.discrimantor(output_I_o)

        vgg_ft_I_o = self.vgg16_extractor(I_o)

        final_loss = (
            self.model_config['muli_level_loss_paramter'] * self.muli_level_loss +
            self.model_config['reconstruction_loss_parameter'] * self.reconstruction_loss(input_I, output_I_o) +
            self.model_config['adversal_loss_paramter'] * self.adversal_loss(pred_I_o, True)
        )

        return final_loss