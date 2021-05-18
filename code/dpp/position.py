import numpy as np
import torch.nn as nn
import torch


def positional_encoding_vector(marks_dim, max_seqlen, dtype=np.float32):

    encoded_vec = np.array([
        pos/np.power(10000, 2*i/marks_dim) for pos in range(max_seqlen) for i in range(marks_dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return encoded_vec.reshape([max_seqlen, marks_dim])