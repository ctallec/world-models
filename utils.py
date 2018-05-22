""" Utilities """
import math
import torch
import numpy as np

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

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename and in best_filename if is_best """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
