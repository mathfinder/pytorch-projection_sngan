import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from source.miscs.random_samples import sample_random, sample_categorical

def loss_dcgan_dis(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss

def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss

def loss_hinge_dis(dis_fake, dis_real):
    loss = torch.mean(F.relu(1. - dis_real))
    loss += torch.mean(F.relu(1. + dis_fake))
    return loss

def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss

