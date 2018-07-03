import torch
import torch.nn as nn
import torch.nn.functional as F
from gen_models.resblocks import Block
from torch.autograd import Variable
from source.miscs.random_samples import sample_categorical, sample_random


class ResNetGenerator(nn.Module):
    def __init__(self, ch=64, ch_out=3, dim_z=128, bottom_width=4, activation=F.relu, n_classes=0, distribution="normal"):
        super(ResNetGenerator, self).__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes

        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 16)
        self.block2 = Block(ch * 16, ch * 16, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = Block(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = Block(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block6 = Block(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b7 = nn.BatchNorm2d(ch)
        self.l7 = nn.Conv2d(ch, ch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, batchsize=64, z=None, y=None):
        if z is None:
            z = Variable(sample_random(self.dim_z, batchsize, self.distribution))
        if y is None:
            y = Variable(
                sample_categorical(self.n_classes, z.size()[0], distribution="uniform").type_as(z)) if self.n_classes > 0 else None

        if (y is not None) and z.size()[0] != y.size()[0]:
            raise Exception('z.shape[0] != y.shape[0], z.shape[0]={}, y.shape[0]={}'.format(z.shape[0], y.shape[0]))
        h = z
        h = self.l1(h)
        h = h.reshape((h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, y) if self.n_classes > 0 else self.block2(h)
        h = self.block3(h, y) if self.n_classes > 0 else self.block3(h)
        h = self.block4(h, y) if self.n_classes > 0 else self.block4(h)
        h = self.block5(h, y) if self.n_classes > 0 else self.block5(h)
        h = self.block6(h, y) if self.n_classes > 0 else self.block6(h)
        h = self.b7(h)
        h = self.activation(h)
        h = F.tanh(self.l7(h))

        return h