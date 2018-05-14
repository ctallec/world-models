""" Some data loading utilities """
from os import listdir
from os.path import join, isdir
import torch
import numpy as np

class RolloutSequenceDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    """ Encapsulate rollouts """
    def __init__(self, root, transform):
        self.transform = transform
        self.files = {}

        i = 0
        for sd in listdir(root):
            subdir = join(root, sd)
            if isdir(subdir):
                for ssd in listdir(subdir):
                    dfile = join(subdir, ssd)
                    self.files[i] = dfile
                    i += 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        wrapped = np.load(self.files[i])
        data = [wrapped[key].astype(np.float32) for key in
                ('actions', 'observations', 'rewards', 'terminals')]
        data[1] = self.transform(data[1])
        return data
