"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import torch.nn as nn

class MDRNN(nn.Module):
    """ MDRNN model """
    def __init__(self, latent_size, action_size, hidden_size):
        super().__init__()
        self.latent_size = latent_size
        self.action_size = action_size
        self.hidden_size = hidden_size

    def forward(self, *inputs):
        """ ONE STEP forward """
        pass
