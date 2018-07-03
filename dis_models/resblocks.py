import math
import torch
from torch.nn import functional as F
import torch.nn as nn
from source.modules.sn_conv2d import SNConv2d

def _downsample(x):
    # todo
    return F.avg_pool2d(x, 2)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(Block, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = SNConv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = SNConv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        if self.learnable_sc:
            self.c_sc = SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.c1 = SNConv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = SNConv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
