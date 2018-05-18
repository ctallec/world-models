""" Test gmm loss """
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm
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
        indices = cat_dist.sample((n_samples,)).long()
        rands = torch.randn(n_samples, 2)

        samples = means[indices] + rands * stds[indices]

        class _model(nn.Module):
            def __init__(self, gaussians):
                super().__init__()
                self.means = nn.Parameter(torch.Tensor(1, gaussians, 2).normal_())
                self.pre_stds = nn.Parameter(torch.Tensor(1, gaussians, 2).normal_())
                self.pi = nn.Parameter(torch.Tensor(1, gaussians).normal_())

            def forward(self, *inputs):
                return self.means, torch.exp(self.pre_stds), f.softmax(self.pi, dim=1)

        model = _model(3)
        optimizer = torch.optim.Adam(model.parameters())

        iterations = 100000
        log_step = iterations // 10
        pbar = tqdm(total=iterations)
        cum_loss = 0
        for i in range(iterations):
            batch = samples[torch.LongTensor(128).random_(0, n_samples)]
            m, s, p = model.forward()
            loss = gmm_loss(batch, m, s, p)
            cum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str("avg_loss={:10.6f}".format(
                cum_loss / (i + 1)))
            pbar.update(1)
            if i % log_step == log_step - 1:
                print(m)
                print(s)
                print(p)
