import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import cuda
import torch.nn as nn
import torch.nn.modules as modules
from source.functional.max_sv import max_singular_value

class SNLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 use_gamma=False, Ip=1, factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNLinear, self).__init__(in_features, out_features, bias)

        # todo
        # self.u = np.random.normal(size=(1, out_features)).astype(dtype="f")
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())
        # self.u = Parameter(torch.Tensor(1, out_features).normal_())
        # self.register_persistent('u')

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
        if self.training :
            # Update estimated 1st singular vector
            self.u[:] = _u
        if hasattr(self, 'gamma'):
            return self.gamma.expand_as(self.weight) * self.weight / sigma
        else:
            return self.weight / sigma

    # todo
    def reset_parameters(self):
        super(SNLinear, self).reset_parameters()
        if self.use_gamma:
            _, s, _ = np.linalg.svd(self.weight.data)
            # todo
            self.gamma = Parameter(torch.Tensor(s[0]).reshape((1, 1)))


    def forward(self, input):

        return F.linear(input, self.W_bar, self.bias)
