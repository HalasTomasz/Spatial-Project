import torch
import torch.nn as nn
from torchvision import models

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
