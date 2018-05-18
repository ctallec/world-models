""" Some data examination """
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.utils import RolloutSequenceDataset

dataloader = DataLoader(
    RolloutSequenceDataset(
        root='datasets/carracing', seq_len=256,
        transform=lambda x: x, buffer_size=10),
    batch_size=1)

monitor = plt.imshow(np.zeros((64, 64, 3)))
for data in dataloader:
    obs = data[0].numpy().unsqueeze()
    monitor.set_data(obs)
    plt.pause(.1)
