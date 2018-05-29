""" Test data loading utilities """
import time
import unittest
import torch
import numpy as np
from torchvision import transforms
from data.loaders import RolloutSequenceDataset
from data.loaders import RolloutObservationDataset

class TestData(unittest.TestCase):
    """ Test data loading """
    def test_rollout_data(self):
        """ Test rollout sequence dataset """
        transform = transforms.Lambda(lambda x: np.transpose(x, (0, 3, 1, 2)))
        dataset = RolloutSequenceDataset('datasets/carracing', 32, transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                             num_workers=8)
        dataset.load_next_buffer()
        init_time = time.time()
        for i, data in enumerate(loader):
            if i == 150:
                self.assertEqual(data[0].size(), torch.Size([8, 32, 3, 96, 96]))
                break
        end_time = time.time()
        print("WallTime: {}s".format(end_time - init_time))

    def test_observation_data(self):
        """ Test observation data """
        transform = transforms.ToTensor()
        dataset = RolloutObservationDataset('datasets/carracing', transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
                                             num_workers=8)
        dataset.load_next_buffer()
        init_time = time.time()
        for i, data in enumerate(loader):
            if i == 150:
                self.assertEqual(data.size(), torch.Size([32, 3, 96, 96]))
                break
        end_time = time.time()
        print("WallTime: {}s".format(end_time - init_time))
