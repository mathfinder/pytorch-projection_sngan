import argparse
import os
import numpy as np
import math

from torch.backends import cudnn

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from dis_models.snresnet import SNResNetProjectionDiscriminator
from gen_models.resnet import ResNetGenerator
from updater import *

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', default=[1])
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_classes', type=int, default=10, help='number of image classes')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
opt = parser.parse_args()
print(opt)


if len(opt.device_ids) > 0:
    torch.cuda.set_device(opt.device_ids[0])
# For fast training
# cudnn.benchmark = True

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Loss function
# adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = ResNetGenerator(ch_out=opt.channels, n_classes=opt.n_classes)
discriminator = SNResNetProjectionDiscriminator(ch_in=opt.channels, n_classes=opt.n_classes)

if cuda:
    generator.cuda()
    discriminator.cuda()
    # adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs('data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    # Batch iterator
    data_iter = iter(dataloader)

    for i in range(len(data_iter) // opt.n_critic):
        # Train discriminator for n_critic times
        for _ in range(opt.n_critic):
            (imgs, clss) = data_iter.next()
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(-1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            clss = clss.cuda()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images

            fake_imgs = generator(z=z, y=clss)

            real_validity = discriminator(real_imgs, clss)
            fake_validity = discriminator(fake_imgs, clss)

            d_loss = loss_hinge_dis(fake_validity, real_validity)
            d_loss.backward()

            optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z=z, y=clss)

        # Loss measures generator's ability to fool the discriminator
        # Train on fake images
        gen_validity = discriminator(gen_imgs, clss)
        g_loss = loss_hinge_gen(gen_validity)
        g_loss.backward()
        optimizer_G.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs,
                                                        batches_done % len(dataloader), len(dataloader),
                                                        d_loss.data[0], gen_validity.data[0]))

        if batches_done % opt.sample_interval == 0:
            name_x = str(clss[:25].cpu().numpy())
            save_image(gen_imgs.data[:25], 'images/'+ name_x +'%d.png' % batches_done, nrow=5, normalize=True)
        batches_done += 1