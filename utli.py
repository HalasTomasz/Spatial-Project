import torch
import torch.nn as nn
from torchvision import models


def initialize_weights(module, init_type='normal', gain=0.02):
    """
    Initialize weights for a given module.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if init_type == 'normal':
            nn.init.normal_(module.weight, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(module.weight, gain=gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(module.weight, gain=gain)
        else:
            raise ValueError(f"Unknown initialization method: {init_type}")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, gain)
        nn.init.constant_(module.bias, 0.0)


def init_net(net, init_type='normal', gain=0.02, device='cpu'):
    """
    Initialize and move the network to the specified device.
    """
    net.apply(lambda m: initialize_weights(m, init_type, gain))
    return net.to(device)

def gram_matrix(feat):
    (batch, ch, h, w) = feat.size()
    feat = feat.view(batch, ch, h*w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


### VGG Feature Extractor ####


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        # Load pre-trained VGG16 model
        vgg16 = models.vgg16(pretrained=True)

        # Define feature extraction layers
        self.enc_1 = nn.Sequential(*vgg16.features[:5])  # First few layers
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])  # Middle layers
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])  # Last layers of interest

        for layer in [self.enc_1, self.enc_2, self.enc_3]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the feature extraction layers.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W)

        Returns:
            List[torch.Tensor]: List of feature maps from set of layers.
        """
    
        features = []

        x = self.enc_1(x)
        features.append(x)

        x = self.enc_2(x)
        features.append(x)

        x = self.enc_3(x)
        features.append(x)

        return features
