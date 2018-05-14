""" Test data loading utilities """
import unittest
import torch
from torchvision import transforms
from data.utils import RolloutSequenceDataset

class TestData(unittest.TestCase):
    """ Test data loading """
    def test_rollout_data(self):
        """ Test rollout sequence dataset """
        transform = transforms.Lambda(lambda x: x.transpose(0, 3, 1, 2) / 255)
        dataset = RolloutSequenceDataset('datasets/carracing', transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                             num_workers=4)
        for data in loader:
            for k in range(4):
                print(data[k].size())
            break
