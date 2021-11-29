import pandas as pd
import numpy as np
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
import matplotlib.pyplot as plt
#from IPython.display import HTML
from dcgan import Generator, Discriminator
from ewc import EWC
import pathlib as plib
import time
from datetime import datetime
from torchvision.utils import save_image
import subprocess
import re
import json
import lpips

N_SAMPLES = 100

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
        result_dict= {'IS': is_mean,'IS_std': is_std, 'KID': kid, 'KID_std' : kid_std, 'FID': fid}
    except:
        print(output)
        result_dict = {}
    return result_dict



def generate_samples(sample_img_noise, generator_fpath):

    netG = Generator(n_gpu).to(device)

    netG.load_state_dict(
        torch.load(generator_fpath, map_location=torch.device('cpu')))


    sample_img_output = netG(sample_img_noise).to('cpu')
    sample_img_list = torch.split(sample_img_output, 1, dim=0)

    for f in work_dir.glob('*'):
        if f.is_file():
            f.unlink()

    counter = 0
    for img in sample_img_list:
        save_image(img.squeeze(),
                   work_dir / f"Sample output {counter}.jpg")
        counter += 1
    return sample_img_list


def compute_lpips(data_fpath, sample_img_list):

    comp_img_dataset = dset.ImageFolder(
        root=data_fpath,
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ]))

    # Create the dataloader
    comp_img_loader = torch.utils.data.DataLoader(
        comp_img_dataset,
        batch_size=N_SAMPLES,
        shuffle=False,
        num_workers=2)
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

    return {'LPIPS': avg, 'LPIPS_std': stderr}


loss_fn_alex = lpips.LPIPS(net='alex')

work_dir = plib.Path.cwd()/"sample_working_dir"
work_dir.mkdir(exist_ok=True)

device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

manualSeed = 999
torch.manual_seed(manualSeed)

sample_img_noise = torch.randn(N_SAMPLES,N_SAMPLES,1,1,device=device)


datasets = ['CelebA_Bald','CelebA_Eyeglasses','CelebA_Bangs', '100-shot-obama', 'AnimalFace-cat']

i = 0
result_dict = {}
for dataset_name in datasets:
    for reg in ['ewc', 'no_ewc']:
        print(dataset_name, reg)
        generator_fpath = f'saved_model/Run_{i}_netG.pt'
        data_fpath = f'data/{dataset_name}_size_64'


        sample_img_list = generate_samples(sample_img_noise, generator_fpath)

        lpips_dict = compute_lpips(data_fpath, sample_img_list)

        lpips_dict = compute_lpips(data_fpath, sample_img_list)



        fid_kid_is_dict = get_fid_kid(data_fpath, work_dir, n_gpu)

        result_dict[f"{dataset_name}_{reg}"] = {**lpips_dict, **fid_kid_is_dict}
        i+=1

result_dict





index = [
    [d for d in datasets for _ in (0, 1)],
    ['With EWC', 'No EWC']*5,
]

index
df = pd.DataFrame(result_dict.values(), index=index)

df
print(df.to_latex(multirow=True))
