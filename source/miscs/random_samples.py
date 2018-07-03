import numpy as np
import torch

def sample_random(dim, batchsize, distribution='normal'):
    if distribution == "normal":
        return torch.Tensor(batchsize, dim).normal_()
    elif distribution == "uniform":
        return torch.Tensor(batchsize, dim).uniform_()
    else:
        raise NotImplementedError


def sample_categorical(n_cat, batchsize, distribution='uniform'):
    if distribution == 'uniform':
        return torch.Tensor(batchsize).random_(0, n_cat)
    else:
        raise NotImplementedError