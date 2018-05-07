"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch.nn as nn

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_size, img_channels, latent_size): # You will probably need more arguments
        super().__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.img_channels = img_channels

    def forward(self, *inputs):
        pass

class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, img_size, img_channels, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.img_channels = img_channels

    def forward(self, *inputs):
        pass
