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
        self.b_size = 32

        dataset = dset.ImageFolder(
            root=dataroot,
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.b_size,
            shuffle=True,
            num_workers=2)

        self.gen_params = {n: p for n, p in self.generator.named_parameters() if p.requires_grad}
        self.disc_params = {n: p for n, p in self.discriminator.named_parameters() if p.requires_grad}
        self.fisher_info_gen, self.fisher_info_disc = self.compute_fisher()

        ## 'star_vars' refers to the pretrained weights.
        ## these should NOT be trainable
        self.gen_star_vars= {}
        for n, p in deepcopy(self.gen_params).items():
            if torch.cuda.is_available():
                p = p.cuda
            self.gen_star_vars[n] = Variable(p.data)

        self.disc_star_vars= {}
        for n, p in deepcopy(self.disc_params).items():
            if torch.cuda.is_available():
                p = p.cuda()
            self.disc_star_vars[n] = Variable(p.data)

    def compute_fisher(self):

        ## do a forward pass to get losses. This is the forward pass from the torch tutorial
        real_label = 1.
        fake_label = 0.
        ### TODO: pass this in as arg
        nz = 100

        criterion = nn.BCELoss(reduction = 'none')

        ### these dicts will store the fisher info for each weight
        '''fisher_generator = {}

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

        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]

        ## just estimate using one batch of data
        #return fisher_generator#, fisher_discriminator
        names = [n for n,p in self.generator.named_parameters()]

        return {n: f.detach() for n, f in zip(names, fisher_diagonals)}'''

        fisher_generator = {}
        fisher_discriminator = {}

        gen_lls = []
        disc_lls = []
        for i, data in enumerate(self.dataloader, 0):

            real_cpu = data[0].to(self.device)
            noise = torch.randn(self.b_size, nz, 1, 1, device=self.device)
            fake = self.generator(noise)
            dat = torch.cat((real_cpu, fake))
            label = torch.cat((torch.full((self.b_size, ),
                               real_label,
                               dtype=torch.float,
                               device=self.device),
                               torch.full((self.b_size, ),
                               fake_label,
                               dtype=torch.float,
                               device=self.device)))

            # Forward pass real batch through D
            output = self.discriminator(dat).view(-1)
            # Calculate loss on all-real batch
            ## comment in the ewc penalty line if you want to incorporate ewc
            #errD_real = criterion(
            #    output, label)
            predictions_real = output[:self.b_size]
            predictions_fake = 1-output[self.b_size:]
            output = -torch.log(torch.cat((predictions_real, predictions_fake)))

            disc_lls.append(output)

            if len(disc_lls) >= self.sample_size // self.b_size:
                break


        disc_lls = torch.cat(disc_lls).unbind()
        disc_ll_grads = zip(*[autograd.grad(
           l, self.discriminator.parameters(),
           retain_graph=(i < len(disc_lls))
        ) for i, l in enumerate(disc_lls, 1)])

        while True:
            ############################
            # (2) Update G network: maximize log(D(G(z))) + lambda * F_i(theta^* - theta)^2
            ###########################
            noise = torch.randn(self.b_size, nz, 1, 1, device=self.device)

            #label.fill_(
            #    real_label)  # fake labels are real for generator cost
            label = torch.full((self.b_size, ),
                         real_label,
                         dtype=torch.float,
                         device=self.device)

            # Since we just updated D, perform another forward pass of all-fake batch through D
            fake = self.generator(noise)
            # Calculate G's loss based on this output and add EWC regularization term!
            ## get the log likelihoods directly
            output = -torch.log(self.discriminator(fake)).view(-1)
            gen_lls.append(output)

            if len(gen_lls) >= self.sample_size // self.b_size:
                break

        gen_lls = torch.cat(gen_lls).unbind()

        # Calculate gradients for G
        #print(summary(self.generator, (100, 1, 1)))
        gen_ll_grads = zip(*[autograd.grad(
            l, self.generator.parameters(),
            retain_graph=(i < len(gen_lls))
        ) for i, l in enumerate(gen_lls, 1)])

        gen_ll_grads = [torch.stack(gs) for gs in gen_ll_grads]
        gen_fisher_diagonals = [(g ** 2).mean(0) for g in gen_ll_grads]

        disc_ll_grads = [torch.stack(gs) for gs in disc_ll_grads]
        disc_fisher_diagonals = [(g ** 2).mean(0) for g in disc_ll_grads]

        ## just estimate using one batch of data
        #return fisher_generator#, fisher_discriminator
        gen_names = [n for n,p in self.generator.named_parameters()]
        disc_names = [n for n,p in self.discriminator.named_parameters()]

        return ({n: f.detach() for n, f in zip(gen_names, gen_fisher_diagonals)},
                {n: f.detach() for n, f in zip(disc_names, disc_fisher_diagonals)})

    ## compute the regularization term
    def penalty(self, model, gen=True):

        params = model.named_parameters()
        if gen:
            star_vars = self.gen_star_vars
            fisher = self.fisher_info_gen
        else:
            star_vars = self.disc_star_vars
            fisher = self.fisher_info_disc

        loss = 0
        #print(star_vars['main.7.weight'])
        for n, p in params:
            penalty = fisher[n] * (p - star_vars[n]) ** 2
            loss += penalty.sum()

        #print(loss)
        return loss

    '''def get_avg_deltas(self, model):

        avg_deltas = {}
        for m in model.modules():
            if isinstance(module, nn.Sequential):
                continue
            change = []
            for n, p in m.named_parameters:
                change.append(p - star_vars[n])
            avg_deltas[m] = torch.mean(torch.cat(change))

        return avg_deltas'''
