import torch
import torch.nn as nn
import torch.nn.functional as F
from source.modules.sn_conv2d import SNConv2d
from source.modules.sn_linear import SNLinear
from source.modules.sn_embedding import SNEmbedding
from dis_models.resblocks import  Block, OptimizedBlock

class SNResNetProjectionDiscriminator(nn.Module):
    def __init__(self, ch=64, ch_in=3, n_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation

        self.block1 = OptimizedBlock(ch_in, ch)
        self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
        self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
        self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
        self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
        self.block6 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
        self.l7 = SNLinear(ch * 16, 1)
        if n_classes > 0:
            self.l_y = SNEmbedding(n_classes, ch * 16)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        h = torch.sum(torch.sum(h, 3), 2)  # Global pooling
        output = self.l7(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + torch.sum(w_y * h, dim=1, keepdim=True)
        return output