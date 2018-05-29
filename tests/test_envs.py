"""
Testing environments
(Non visual testing)
"""
import unittest
from envs.simulated_carracing import SimulatedCarracing
from utils.misc import sample_continuous_policy
import numpy as np
import matplotlib.pyplot as plt

FPS = 50

class TestEnvs(unittest.TestCase):
    """ Test environments """
    def test_simulated_carracing(self):
        """ Test simulated Car Racing """
        env = SimulatedCarracing('logs/exp0')
        env.reset()
        seq_len = 1000
        actions = sample_continuous_policy(
            env.action_space, seq_len, 1. / FPS)
        for i in range(seq_len):
            action = actions[i]
            next_obs, reward, terminal = env.step(action)
            env.render()
            print(next_obs.shape, reward)
            if terminal:
                break
