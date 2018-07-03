import math
import torch
from torch.nn import functional as F
import torch.nn as nn
# from source.modules.categorical_conditional_batch_normalization import CategoricalConditionalBatchNormalization
from source.nn.condinstancenorm import CondInstanceNorm2d
def _upsample(x):
    # todo
    h, w = x.shape[2:]
    return F.upsample(x, size=(h * 2, w * 2))

def upsample_conv(x, conv):
    return conv(_upsample(x))

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, n_classes=0):
        super(Block, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = (in_channels != out_channels) or upsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes

        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        if n_classes > 0:
            self.b1 = CondInstanceNorm2d(in_channels, num_labels=n_classes)
            self.b2 = CondInstanceNorm2d(hidden_channels, num_labels=n_classes)
        else:
            self.b1 = nn.BatchNorm2d(in_channels)
            self.b2 = nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def residual(self, x, y=None, z=None, **kwargs):
        h = x
        h = self.b1(h, y) if y is not None else self.b1(h, **kwargs)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y) if y is not None else self.b2(h, **kwargs)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None, z=None, **kwargs):
        return self.residual(x, y, z, **kwargs) + self.shortcut(x)

