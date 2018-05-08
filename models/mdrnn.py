"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def gmm_loss(batch, mus, sigmas, pi, reduce=True):
    """ Computes the gmm loss """
    batch = batch.unsqueeze(1)
    normal_dist = Normal(mus, sigmas)
    probs = torch.sum((torch.exp(normal_dist.log_prob(batch)) * pi), dim=1)
    log_prob = torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob

class MDRNN(nn.Module):
    """ MDRNN model """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.rnn = nn.LSTMCell(latents + actions, hiddens)
        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians)


    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
        """ ONE STEP forward """
        in_al = torch.cat([action, latent], dim=1)
        out_rnn, next_hidden = self.rnn(in_al, hidden)
        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents
        mus = out_full[:, :stride].view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(
            out_full[:, stride:2 * stride].view(-1, self.gaussians, self.latents))
        pi = torch.softmax(out_full[:, - self.gaussians:].view(-1, self.gaussians, 1))
        return mus, sigmas, pi, next_hidden
