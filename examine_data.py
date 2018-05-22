""" Some data examination """
import math
import numpy as np
import matplotlib.pyplot as plt

def plot_rollout():
    """ Plot a rollout """
    from torch.utils.data import DataLoader
    from data.utils import RolloutSequenceDataset
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

def sample_continuous_policy(action_space, seq_len, dt):
    """
    Sample a continuous policy. Atm, action_space is supposed
    to be a box environment.
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def plot_continuous_policy():
    """ Plot a rollout using a continuous policy """
    import gym
    from gym.envs.box2d.car_racing import FPS
    episode_length = 1000
    env = gym.make('CarRacing-v0')
    env.reset()

    actions = sample_continuous_policy(env.action_space, episode_length, 1 / FPS)

    for i in range(episode_length):
        env.step(actions[i])
        env.render()

if __name__ == '__main__':
    plot_rollout()
    plot_continuous_policy()
