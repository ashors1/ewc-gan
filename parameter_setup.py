import torch
import argparse


def parameter_setup():
    # Default Setting
    # fine_tune vs. pre-trained
    fine_tune = True
    pre_trained_G_path = "netG_10_epoch_state_dict"
    pre_trained_D_path = "netD_10_epoch_state_dict"

    # data configuration
    dataroot = "./data/AF_Mini"
    batch_size = 4
    image_size = 64
    workers = 2

    # devive setup
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")

    # training setup for DCGAN in general
    D_lr = 0.0002
    D_beta1 = 0.5
    D_beta2 = 0.999

    G_lr = 0.0002
    G_beta1 = 0.5
    G_beta2 = 0.999

    num_epochs = 5

    # training setup for EWC
    ewc_lambda = 0
    ewc_data_root = './data/CelebA'
    ewc_dict = {"ewc_lambda": ewc_lambda, "ewc_data_root": ewc_data_root}

    # containing all relevant parameters
    train_dict = {
        "D_lr": D_lr,
        "D_beta1": D_beta1,
        "D_beta2": D_beta2,
        "G_lr": G_lr,
        "G_beta1": G_beta1,
        "G_beta2": G_beta2,
        "num_epochs": num_epochs,
        "device": device,
        "data_root": dataroot,
        "batch_size": batch_size,
        "ewc_lambda": ewc_lambda,
        "ngpu": ngpu,
        "fine_tune": fine_tune,
        "pre_G": pre_trained_G_path,
        "pre_D": pre_trained_D_path,
        "image_size": image_size,
        "workers": workers
    }

    return train_dict, ewc_dict


def train_arg_parser():
    # TODO: finish configuring arg parser
    pass
