import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import cuda
from torch.nn import Conv2d
import torch.nn.modules as modules
from source.functional.max_sv import max_singular_value

class SNConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, use_gamma=False, Ip=1, factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)


        # todo
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())
        # self.u = self.register_buffer(torch.Tensor(1, out_channels).normal_())
        # self.register_persistent('u')

        self.reset_parameters()

    @property
    def W_bar(self):
        """
        Spectrally Normalized Weight
        :return: 
        """
        W_mat = self.weight.reshape(self.weight.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        # todo
        sigma = sigma.reshape((1, 1, 1, 1)).expand(self.weight.shape)
        if self.training :
            # Update estimated 1st singular vector
            self.u[:] = _u
        if hasattr(self, 'gamma'):
            return self.gamma.expand_as(self.weight) * self.weight / sigma
        else:
            return self.weight / sigma

    # todo
    def reset_parameters(self):
        super(SNConv2d, self).reset_parameters()
        if self.use_gamma:
            W_mat = self.weight.data.reshape(self.weight.shape[0], -1)
            # todo
            _, s, _ = np.linalg.svd(W_mat.numpy())
            self.gamma = Parameter(torch.Tensor([s[0]]).reshape((1, 1, 1, 1)))


    def forward(self, input):
        return F.conv2d(input, self.W_bar, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
