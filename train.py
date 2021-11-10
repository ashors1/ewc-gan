from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
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
#from IPython.display import HTML
from dcgan import Generator, Discriminator
from ewc import EWC
import pathlib as plib
import time
from datetime import datetime
import argparse
from parameter_setup import parameter_setup


# custom weights initialization called on netG and netD
# Specified by DCGAN tutorial at https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def ewc_loss(model):
    #TODO: Implement
    return 0


def train(netG, netD, dataloader, train_dict, ewc_dict):

    ############################################
    ######   Parameter Setup
    ############################################
    nz = 100

    D_lr = train_dict['D_lr']
    D_beta1 = train_dict['D_beta1']
    D_beta2 = train_dict['D_beta2']

    G_lr = train_dict['G_lr']
    G_beta1 = train_dict['G_beta1']
    G_beta2 = train_dict['G_beta2']

    num_epochs = train_dict['num_epochs']

    device = train_dict['device']
    ngpu = train_dict["ngpu"]

    # EWC Setup
    ## dataroot is path to celeba dataset
    ewc_data_root = ewc_dict['ewc_data_root']
    ewc = EWC(ewc_data_root, 1024, netG, netD)
    print('done with initialization')

    lam = ewc_dict["ewc_lambda"]

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    ############################################
    ######   Logging Setup
    ############################################
    log_dir = plib.Path.cwd() / "log"
    log_dir.mkdir(exist_ok=True)

    existing_log_files_versions = [
        int(f.name.replace(".log", "").replace("Run ", ""))
        for f in log_dir.glob('*.log') if f.is_file()
    ]

    if len(existing_log_files_versions) == 0:
        current_version = 0
    else:
        current_version = max(existing_log_files_versions) + 1

    log_file_path = log_dir / f"Run {current_version}.log"

    ############################################
    ######   Loss Function and Optimizer
    ############################################
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    log_img_dict = dict()

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(),
                            lr=D_lr,
                            betas=(D_beta1, D_beta2))
    optimizerG = optim.Adam(netG.parameters(),
                            lr=G_lr,
                            betas=(G_beta1, G_beta2))

    # Create a fixed batch of latent vectors to track progress
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    ## training loop for fine-tuning is identical to pre-training training loop
    ## except we add the EWC penalty to the loss function
    print("Starting Training Loop...")
    # For each epoch

    with log_file_path.open('w', encoding="utf-8") as f_handle:
        ############################
        # Logging
        ###########################
        f_handle.write(f"Training Run {current_version}:\n")
        for k, v in train_dict.items():
            f_handle.write(f" {k} : {v} \n")
        f_handle.write(f"Starting time: {datetime.now()} \n")
        start_time = time.time()

        ############################
        # Training
        ###########################

        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                #### training loop from pytorch tutorial ####

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()

                ### i've been changing niters to adjust how many generator training iterations we do for each discriminator
                ### iteration
                if iters % train_dict['D_update_rate'] == 0:
                    # Format batch
                    real_cpu = data[0].to(device)
                    b_size = real_cpu.size(0)
                    label = torch.full((b_size, ),
                                       real_label,
                                       dtype=torch.float,
                                       device=device)
                    # Forward pass real batch through D
                    output = netD(real_cpu).view(-1)
                    # Calculate loss on all-real batch
                    ## comment in the ewc penalty line if you want to incorporate ewc
                    errD_real = criterion(
                        output, label)  #+ ewc.penalty(netD, gen=False)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    D_x = output.mean().item()

                    ## Train with all-fake batch
                    # Generate batch of latent vectors
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    # Generate fake image batch with G
                    fake = netG(noise)
                    label.fill_(fake_label)
                    # Classify all fake batch with D
                    output = netD(fake.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    ## commend in ewc penalty here too
                    errD_fake = criterion(
                        output, label)  #+ lam * ewc.penalty(netD, gen=False)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    # Update D
                    optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z))) + lambda * F_i(theta^* - theta)^2
                ###########################
                netG.zero_grad()
                label.fill_(
                    real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output and add EWC regularization term!
                #print(ewc.penalty(netG))
                ## ewc penalty for generator
                errG = criterion(output, label)  #+ lam * ewc.penalty(netG)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                msg = '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (
                    epoch, num_epochs, i, len(dataloader), errD.item(),
                    errG.item(), D_x, D_G_z1, D_G_z2)
                if i % 50 == 0:
                    print(msg)

                # writing training stats to log file
                f_handle.write(msg + "\n")

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and
                                          (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = netG(noise).detach().cpu()
                    img_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True))

                ## I addded this to inspect the generated images every 10 iterations
                if iters % 10 == 0:
                    with torch.no_grad():
                        fake = netG(noise).detach().cpu()

                    img_grid = np.transpose(
                        vutils.make_grid(fake, padding=2, normalize=True),
                        (1, 2, 0))

                    log_img_dict[
                        log_dir /
                        f"Run {current_version} Fixed Noise Output at Iter {iters}.png"] = img_grid

                    plt.imshow(img_grid)
                    #plt.imshow(np.transpose(vutils.make_grid(torch.cat((data[0], fake)), padding=2, normalize=True), (1,2,0)))
                    #plt.show()

                iters += 1

        ############################
        # Finish Logging
        ############################
        #
        if train_dict['save']:
            model_dir = plib.Path.cwd() / "saved_model"
            model_dir.mkdir(exist_ok=True)
            torch.save(netD.state_dict(),
                       model_dir / f"Run_{current_version}_netD.pt")
            torch.save(netG.state_dict(),
                       model_dir / f"Run_{current_version}_netG.pt")

        f_handle.write(
            f"Training Finished, total run time {round((start_time - time.time())/60)} minutes."
        )

        for k, v in log_img_dict.items():
            plt.imshow(v)
            plt.savefig(k)

        # plot last image
        plt.subplot(1, 2, 2)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.savefig(log_dir /
                    f"Run {current_version} Fixed Noise Output at Iter -1.png")
        #plt.show()

        # plot loss
        plt.figure().clear()
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(log_dir / f"Run {current_version} loss.png")


if __name__ == '__main__':

    train_dict, ewc_dict = parameter_setup()
    ############################################
    ###### Model Initialization
    ############################################
    if train_dict['pretrain']:
        # pre-training
        netG = Generator(train_dict['ngpu']).to(train_dict['device'])

        # Create the Discriminator
        netD = Discriminator(train_dict['ngpu']).to(train_dict['device'])

        # Handle multi-gpu if desired
        if (train_dict['device'].type == 'cuda') and (train_dict['ngpu'] > 1):
            netG = nn.DataParallel(netG, list(range(train_dict['ngpu'])))
            netD = nn.DataParallel(netD, list(range(train_dict['ngpu'])))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.02.
        netG.apply(weights_init)
        netD.apply(weights_init)

    else:
        netG = Generator(train_dict['ngpu']).to(train_dict['device'])
        ### map_location for running on cpu
        netG.load_state_dict(
            torch.load(train_dict['pre_G'], map_location=torch.device('cpu')))

        netD = Discriminator(train_dict['ngpu']).to(train_dict['device'])
        netD.load_state_dict(
            torch.load(train_dict['pre_D'], map_location=torch.device('cpu')))

    ############################################
    ###### Data Loader Initialization
    ############################################

    dataset = dset.ImageFolder(
        root=train_dict['data_root'],
        transform=transforms.Compose([
            transforms.Resize(train_dict['image_size']),
            transforms.CenterCrop(train_dict['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_dict['batch_size'],
        shuffle=True,
        num_workers=train_dict['workers'])

    ############################################
    ###### Calling Training
    ############################################

    train(netG, netD, dataloader, train_dict, ewc_dict)
