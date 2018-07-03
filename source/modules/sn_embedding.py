import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import cuda
import torch.nn as nn
from source.functional.max_sv import max_singular_value

class SNEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None, Ip=1, factor=None):

        super(SNEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse, _weight=_weight)
        self.Ip = Ip
        self.factor = factor

        # todo
        self.register_buffer('u', torch.Tensor(1, num_embeddings).normal_())

        self.reset_parameters()

    @property
    def W_bar(self):
        """
        Spectrally Normalized Weight
        :return: 
        """
        # W_mat = self.weight.reshape(self.weight.shape[0], -1)
        sigma, _u, _ = max_singular_value(self.weight, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        # todo
        sigma = sigma.reshape((1, 1)).expand_as(self.weight)
        self.u[:] = _u

        return self.weight / sigma

    # todo


    def forward(self, input):
        return F.embedding(input, self.W_bar, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq,
                           self.sparse)
