""" Some data examination """
import numpy as np
import matplotlib.pyplot as plt

def plot_rollout():
    """ Plot a rollout """
    from torch.utils.data import DataLoader
    from data.loaders import RolloutSequenceDataset
    dataloader = DataLoader(
        RolloutSequenceDataset(
            root='datasets/carracing', seq_len=900,
            transform=lambda x: x, buffer_size=10,
            train=False),
        batch_size=1, shuffle=True)

    dataloader.dataset.load_next_buffer()

    # setting up subplots
    plt.subplot(2, 2, 1)
    monitor_obs = plt.imshow(np.zeros((64, 64, 3)))
    plt.subplot(2, 2, 2)
    monitor_next_obs = plt.imshow(np.zeros((64, 64, 3)))
    plt.subplot(2, 2, 3)
    monitor_diff = plt.imshow(np.zeros((64, 64, 3)))

    for data in dataloader:
        obs_seq = data[0].numpy().squeeze()
        action_seq = data[1].numpy().squeeze()
        next_obs_seq = data[-1].numpy().squeeze()
        for obs, action, next_obs in zip(obs_seq, action_seq, next_obs_seq):
            monitor_obs.set_data(obs)
            monitor_next_obs.set_data(next_obs)
            monitor_diff.set_data(next_obs - obs)
            print(action)
            plt.pause(.01)
        break

if __name__ == '__main__':
    plot_rollout()
