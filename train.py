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
import lpips
from torchvision.utils import save_image
import subprocess
import re
import json
# custom weights initialization called on netG and netD
# Specified by DCGAN tutorial at https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# Adds 2 GAN Hack methods for stabilizing training/handicapping the discriminator using
# one-sided label smoothing and instance noise:
# https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_fid_kid(og_dir, sample_dir, ngpu):
    og_dir = plib.Path.cwd() / plib.Path(og_dir)
    for f_handle in og_dir.glob("*"):
        if f_handle.is_dir():
            f = f_handle
    og_dir = f

    run_str = f"fidelity{' --gpu 0' if ngpu > 0 else ''} --isc --fid --kid --input1 {str(sample_dir)} --input2 {str(og_dir)} --kid-subset-size 100"

    run_arglist = run_str.split(" ")
    process = subprocess.run(run_arglist,
                             check=True,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
    output = process.stdout

    try:
        is_mean = float(
            re.search('(?<=inception_score_mean:).*(?=\n)', output).group(0))
        is_std = float(
            re.search('(?<=inception_score_std:).*(?=\n)', output).group(0))
        fid = float(
            re.search('(?<=frechet_inception_distance:).*(?=\n)',
                      output).group(0))
        kid = float(
            re.search('(?<=kernel_inception_distance_mean:).*(?=\n)',
                      output).group(0))
        kid_std = float(
            re.search('(?<=kernel_inception_distance_std:).*(?=\n)',
                      output).group(0))
        result_list = [is_mean, is_std, fid, kid, kid_std]
    except:
        print(output)
        result_list = []
    return result_list


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
    ewc = EWC(ewc_data_root, 32, netG, netD)
    print('done with initialization')

    d_lam = ewc_dict["D_ewc_lambda"]
    g_lam = ewc_dict["G_ewc_lambda"]

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

    summary_dir = log_dir / "training_summary"
    summary_dir.mkdir(exist_ok=True)

    intermediate_img_dir = log_dir / "intermediate_img"
    intermediate_img_dir.mkdir(exist_ok=True)

    final_img_dir = log_dir / "final_img"
    final_img_dir.mkdir(exist_ok=True)

    work_dir = log_dir / "score_working_dir"
    work_dir.mkdir(exist_ok=True)

    metrics_dir = log_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    existing_log_files_versions = [
        int(f.name.replace(".log", "").replace("Run ", ""))
        for f in summary_dir.glob('*.log') if f.is_file()
    ]

    if len(existing_log_files_versions) == 0:
        current_version = 0
    else:
        current_version = max(existing_log_files_versions) + 1

    log_file_path = summary_dir / f"Run {current_version}.log"
    metrics_file_path = metrics_dir / f"Run {current_version}.json"

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

    # setup LPIPS loss function
    loss_fn_alex = lpips.LPIPS(net='alex')
    lpips_dict = dict()
    fkid_dict = dict()

    ## training loop for fine-tuning is identical to pre-training training loop
    ## except we add the EWC penalty to the loss function
    print("Starting Training Loop...")
    # For each epoch

    netD.train()
    netG.train()

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

                    #one-sided label smoothing
                    if train_dict['label_smoothing_p'] != 0:
                        flip_idxs = torch.bernoulli((1-train_dict['label_smoothing_p'])*torch.ones(b_size))

                    # #instance noise
                    if train_dict['instance_noise_sigma'] != 0:
                        sigma_anneal = (num_epochs - epoch)/num_epochs*train_dict['instance_noise_sigma']
                        instance_noise_real = sigma_anneal *torch.randn(size = real_cpu.size())
                        instance_noise_fake = sigma_anneal *torch.randn(size = real_cpu.size())
                    else:
                        instance_noise_real = 0
                        instance_noise_fake = 0

                    # Forward pass real batch through D
                    output = netD(real_cpu + instance_noise_real).view(-1)
                    # Calculate loss on all-real batch
                    ## comment in the ewc penalty line if you want to incorporate ewc
                    errD_real = criterion(
                        output, label) + d_lam * ewc.penalty(netD, gen=False)
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
                    output = netD(fake.detach() + instance_noise_fake).view(-1)
                    # Calculate D's loss on the all-fake batch
                    ## commend in ewc penalty here too
                    errD_fake = criterion(
                        output, label) + d_lam * ewc.penalty(netD, gen=False)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake + d_lam * ewc.penalty(netD, gen=False)
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
                ## ewc penalty for generator
                ewc_penalty = ewc.penalty(netG)
                errG = criterion(output, label) + g_lam * ewc_penalty

                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                msg = '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tEWC: %.4g' % (
                    epoch, num_epochs, i, len(dataloader), errD.item(),
                    errG.item(), D_x, D_G_z1, D_G_z2, ewc_penalty)
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
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True))

                ## I addded this to inspect the generated images every 10 iterations
                if iters % train_dict['img_freq'] == 0:
                    with torch.no_grad():
                        fake = netG(noise).detach().cpu()

                    img_grid = np.transpose(
                        vutils.make_grid(fake, padding=2, normalize=True),
                        (1, 2, 0))

                    log_img_dict[
                        intermediate_img_dir /
                        f"Run {current_version} Fixed Noise Output at Iter {iters}.png"] = img_grid

                    plt.imshow(img_grid)

                # score
                if train_dict['score_freq'] == 0:
                    iters += 1
                    continue
                elif ((train_dict['score_freq'] != -1) and (iters % train_dict['score_freq'] == 0) and (iters != 0)) or \
                    ((train_dict['score_freq'] == -1) and (iters == num_epochs*len(dataloader) - 1)): #last iteration

                    sample_img_noise = torch.randn(100,
                                                   nz,
                                                   1,
                                                   1,
                                                   device=device)
                    sample_img_output = netG(sample_img_noise).to('cpu')
                    sample_img_list = torch.split(sample_img_output, 1, dim=0)
                    comp_img_dataset = dset.ImageFolder(
                        root=train_dict['data_root'],
                        transform=transforms.Compose([
                            transforms.Resize(train_dict['image_size']),
                            transforms.CenterCrop(train_dict['image_size']),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5)),
                        ]))
                    # Create the dataloader
                    comp_img_loader = torch.utils.data.DataLoader(
                        comp_img_dataset,
                        batch_size=dataset.__len__(),
                        shuffle=False,
                        num_workers=train_dict['workers'])
                    comp_imgs = next(iter(comp_img_loader))[0].cpu()
                    comp_img_list = torch.split(comp_imgs, 1, dim=0)

                    #
                    LPIPS_score_list = []
                    for img1 in sample_img_list:
                        for img2 in comp_img_list:
                            LPIPS_score_list.append(
                                loss_fn_alex(img1, img2).squeeze().item())

                    avg = np.mean(LPIPS_score_list)
                    stderr = np.std(np.array(LPIPS_score_list)) / np.sqrt(
                        len(LPIPS_score_list))

                    lpips_dict[iters] = (avg, stderr)

                    # FID KID Calculation
                    # erasing all files in the work folder

                    for f in work_dir.glob('*'):
                        if f.is_file():
                            f.unlink()

                    counter = 0
                    for img in sample_img_list:
                        save_image(img.squeeze(),
                                   work_dir / f"Sample output {counter}.png")
                        counter += 1
                    fid_kid_result = get_fid_kid(train_dict['data_root'],
                                                 work_dir,
                                                 train_dict['ngpu'])
                    fkid_dict[iters] = fid_kid_result
               
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

        # write LPIPS score

        # for k, v in lpips_dict.items():
        #     f_handle.write(
        #         f"\n#At iteration {k}, the average LPIPS score is {v[0]} and the standerr is {v[1]}"
        #     )
        metric_names = ["is", "is_std", "fid", "kid", "kid_std", "lpips", "lpips_std"]
        metrics_list = []
        for k, v in fkid_dict.items():
            v_all = v + list(lpips_dict[k])
            metrics_list.append(v_all)
            f_handle.write(
                f"\n#At iteration {k} the {','.join(metric_names)} is {v_all}"
            )

        #write metrics to json with params
        metrics_dict = {k: v for k, v in train_dict.items() if k != 'device'}
        for m_i, m_list in enumerate (zip(*metrics_list)):
            metrics_dict[metric_names[m_i]] = m_list

        with open(metrics_file_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        for k, v in log_img_dict.items():
            plt.imshow(v)
            plt.savefig(k)

        # plot last image
        plt.subplot(1, 2, 2)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        # plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.savefig(final_img_dir /
                    f"Run {current_version} Fixed Noise Output at Iter -1.png",
                    bbox_inches='tight',
                    transparent=True,
                    pad_inches=.1)
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
        plt.savefig(final_img_dir / f"Run {current_version} loss.png")


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

    torch.manual_seed(999)
    n_shots = train_dict['num_shots'] 
    if n_shots != -1:
        subset = torch.torch.randperm(len(dataset))[:n_shots]
        dataset = torch.utils.data.Subset(dataset, subset)


        log_dir = plib.Path.cwd() / "log"
        log_dir.mkdir(exist_ok=True)
 
        summary_dir = log_dir / "training_summary"
        summary_dir.mkdir(exist_ok=True)
        existing_log_files_versions = [
           int(f.name.replace(".log", "").replace("Run ", ""))
           for f in summary_dir.glob('*.log') if f.is_file()
        ]

        if len(existing_log_files_versions) == 0:
            current_version = 0
        else:
            current_version = max(existing_log_files_versions) + 1

        few_shot_dir = log_dir / "few_shot_datasets"
        few_shot_dir.mkdir(exist_ok=True)
        subset_dir = few_shot_dir / f"run_{current_version}_{n_shots}_shots" 
        subset_dir.mkdir(exist_ok=True)

        for idx, data in enumerate(dataset):
            data = data[0]
            vutils.save_image(data, subset_dir / 'img_{}.png'.format(idx), normalize=True)


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
