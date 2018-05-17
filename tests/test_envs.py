"""
Testing environments
(Non visual testing)
"""
import unittest
from envs.simulated_carracing import SimulatedCarracing

class TestEnvs(unittest.TestCase):
    """ Test environments """
    def test_simulated_carracing(self):
        """ Test simulated Car Racing """
        env = SimulatedCarracing('logs/exp0')
        env.reset()
        while True:
            action = env.action_space.sample()
            next_obs, reward, terminal = env.step(action)
            print(next_obs.shape, reward)
            if terminal:
                break
