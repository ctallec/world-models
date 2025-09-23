# data/carracing.py
import argparse
import numpy as np
import gymnasium as gym
from os.path import join, exists
from utils.misc import sample_continuous_policy

def generate_data(rollouts, data_dir, noise_type):  # pylint: disable=R0914
    assert exists(data_dir), "The data directory does not exist..."
    env = gym.make("CarRacing-v3", render_mode=None)  # headless frame capture [Gymnasium]
    seq_len = 1000

    for i in range(rollouts):
        obs, info = env.reset()  # new API
        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1.0 / 50)

        s_rollout, r_rollout, d_rollout = [], [], []
        t = 0
        while True:
            action = a_rollout[t]
            t += 1
            s, r, terminated, truncated, info = env.step(action)  # new API
            done = terminated or truncated
            s_rollout.append(s)
            r_rollout.append(r)
            d_rollout.append(done)
            if done or t >= seq_len:
                print(f"> End of rollout {i}, {len(s_rollout)} frames...")
                np.savez(
                    join(data_dir, f"rollout_{i}"),
                    observations=np.array(s_rollout),
                    rewards=np.array(r_rollout),
                    actions=np.array(a_rollout[:len(s_rollout)]),
                    terminals=np.array(d_rollout),
                )
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.', default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)
