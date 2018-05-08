""" Test gmm loss """
import unittest
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
from models.mdrnn import gmm_loss

class TestGMM(unittest.TestCase):
    """ Test GMMs """
    def test_gmm_loss(self):
        """ Test case 1 """
        n_samples = 10000

        means = torch.Tensor([[0., 0.],
                              [1., 1.],
                              [-1., 1.]])
        stds = torch.Tensor([[.03, .05],
                             [.02, .1],
                             [.1, .03]])
        pi = torch.Tensor([.2, .3, .5])

        cat_dist = Categorical(pi)
        indices = cat_dist.sample_n(n_samples).long()
        rands = torch.randn(n_samples, 2)

        samples = means[indices] + rands * stds[indices]

        class _model(nn.Module):
            def __init__(self, gaussians):
                super().__init__()
                self.means = nn.Parameter(torch.Tensor(1, gaussians, 2).normal_())
                self.pre_stds = nn.Parameter(torch.Tensor(1, gaussians, 2).normal_())
                self.pi = nn.Parameter(torch.Tensor(1, gaussians, 1).normal_())

            def forward(self, *inputs):
                return self.means, torch.exp(self.pre_stds), f.softmax(self.pi, dim=1)

        model = _model(10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        for i in range(100000):
            batch = samples[torch.LongTensor(32).random_(0, n_samples)]
            m, s, p = model.forward()
            loss = gmm_loss(batch, m, s, p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 1000 == 999:
                print(loss.item())
                print(m)
                print(s)
                print(p)
