### code adapted from https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
### and pytorch dcgan tutorial

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

import argparse
import os
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

class EWC(object):

    def __init__(self, dataroot, batch_size, generator: nn.Module, discriminator: nn.Module):

        ## if we're just computing fisher info for generator, no need for the actual training data
        '''image_size = 64
       
        ## this is the celebA dataset. I was computing fisher info for the discriminator based
        ## on both the generated data and the training data but I guess that's not correct
        self.dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)'''

        # Decide which device we want to run on
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        ## number of examples to use to estimate fisher information
        self.bs = batch_size

        self.gen_params = {n: p for n, p in self.generator.named_parameters() if p.requires_grad}
        #self.disc_params = {n: p for n, p in self.discriminator.named_parameters() if p.requires_grad}
        ## estimate the fisher info on a batch of data
        #self.fisher_info_gen, self.fisher_info_disc= self.compute_fisher()
        self.fisher_info_gen = self.compute_fisher()
   
        ## 'star_vars' refers to the pretrained weights.
        ## these should NOT be trainable
        self.gen_star_vars= {}
        for n, p in deepcopy(self.gen_params).items():
            if torch.cuda.is_available():
                p = p.cuda
            self.gen_star_vars[n] = Variable(p.data)

    def compute_fisher(self):

        ## do a forward pass to get losses. This is the forward pass from the torch tutorial
        real_label = 1.
        fake_label = 0.
        ### TODO: pass this in as arg
        nz = 100

        criterion = nn.BCELoss()

        ### these dicts will store the fisher info for each weight
        fisher_generator = {}
        fisher_discriminator = {}

        for n, p in deepcopy(self.gen_params).items():
            #p.requires_grad = False
            p.data.zero_()
            if torch.cuda.is_available():
                p = p.cuda
            fisher_generator[n] = Variable(p.data)

        self.generator.eval()
        self.discriminator.eval()

        #for i, data in enumerate(self.dataloader, 0):
        for _ in range(5):

            ### from paper, it looks like only generator's fisher information is relevant!!
            ### see end of page 4

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.generator.zero_grad()
            ## with this line commented out, dont the training issues. With it in, I do
            noise = torch.randn(self.bs, nz, 1, 1, device=self.device)
            fake = self.generator(noise)

            label = torch.full((self.bs,), real_label, dtype=torch.float, device=self.device)
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()

            ### store gradients in the fisher info dicts
            for n, p in self.generator.named_parameters():
                ### idk why the original implementation divided by bs here...
                ### loss reduction is mean so we should already be dividing by bs in the loss computation
                fisher_generator[n].data += p.grad.data ** 2 #/ self.bs

            fisher_generator = {n: p for n, p in fisher_generator.items()}

            self.generator.train()
            self.discriminator.train()

            ## just estimate using one batch of data
            return fisher_generator#, fisher_discriminator

    ## compute the regularization term
    def penalty(self, model, gen=True):

        params = model.named_parameters()
        star_vars = self.gen_star_vars
        fisher = self.fisher_info_gen

        '''for n, p in params:
            print(p[0][0])
            print(star_vars[n][0][0])
            break'''

        loss = 0
        #print(star_vars['main.7.weight'])
        for n, p in params:
            penalty = fisher[n] * (p - star_vars[n]) ** 2
            loss += penalty.sum()

        #print(loss)
        return loss
