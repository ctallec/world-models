""" Utilities """
import torch

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename and in best_filename if is_best """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
