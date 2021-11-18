### code adapted from https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
### and pytorch dcgan tutorial

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd
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
#from torchsummary import summary

class EWC(object):

    def __init__(self, dataroot, sample_size, generator: nn.Module, discriminator: nn.Module):

        # Decide which device we want to run on
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        ## number of examples to use to estimate fisher information
        self.sample_size = sample_size

        self.gen_params = {n: p for n, p in self.generator.named_parameters() if p.requires_grad}
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

        criterion = nn.BCELoss(reduction = 'none')

        ### these dicts will store the fisher info for each weight
        fisher_generator = {}

        #self.generator.zero_grad()

        ### ADD BATCHING TO SPEED UP COMPUTATION ###
        ## currently batch size is hard coded to 32
        loglikelihoods = []
        for _ in range(100):

            noise = torch.randn(32, nz, 1, 1, device=self.device)
            fake = self.generator(noise)

            label = torch.full((32,), real_label, dtype=torch.float, device=self.device)
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D

            ## get the log likelihoods directly
            output = torch.log(self.discriminator(fake)).view(-1)
            # Calculate G's loss based on this output
            #output = self.discriminator(fake).view(-1)
            #errG = criterion(output, label)
            #print(errG.shape)

            #loglikelihoods.append(
            #    -errG
            #)

            #print(output)
            loglikelihoods.append(output)

            if len(loglikelihoods) >= self.sample_size // 32:
                break

        loglikelihoods = torch.cat(loglikelihoods).unbind()

        # Calculate gradients for G
        #print(summary(self.generator, (100, 1, 1)))
        loglikelihood_grads = zip(*[autograd.grad(
            l, self.generator.parameters(),
            retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])

        '''for gs in loglikelihood_grads:
            for i in gs:
                print(i.shape)
            print()
            print('*****************')'''

        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]

        ## just estimate using one batch of data
        #return fisher_generator#, fisher_discriminator
        names = [n for n,p in self.generator.named_parameters()]

        return {n: f.detach() for n, f in zip(names, fisher_diagonals)}

    ## compute the regularization term
    def penalty(self, model, gen=True):

        params = model.named_parameters()
        star_vars = self.gen_star_vars
        fisher = self.fisher_info_gen

        loss = 0
        #print(star_vars['main.7.weight'])
        for n, p in params:
            penalty = fisher[n] * (p - star_vars[n]) ** 2
            loss += penalty.sum()

        #print(loss)
        return loss
