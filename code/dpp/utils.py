import numpy as np
import torch.nn as nn
import torch

def get_inter_times(arrival_times):
    """Convert arrival times to interevent times."""
    return arrival_times - np.concatenate([[0], arrival_times[:-1]])


def get_arrival_times(inter_times):
    """Convert interevent times to arrival times."""
    return inter_times.cumsum()


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def clamp_preserve_gradients(x, min, max):
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """  (For Attentive Model AMDN - Transformer blocks)
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """
    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval