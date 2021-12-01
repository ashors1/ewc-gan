import pandas as pd
import numpy as np
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
import glob
from collections import defaultdict
pd.options.display.float_format = '{:,.3g}'.format

N_SAMPLES = 100

def get_fid_kid(og_dir, sample_dir, ngpu = 1):
	og_dir = plib.Path.cwd() / plib.Path(og_dir)
	for f_handle in og_dir.glob("*"):
		if f_handle.is_dir():
			f = f_handle
	og_dir = f

	n_kid =len(os.listdir(og_dir))

	run_str = f"fidelity{' --gpu 0' if ngpu > 0 else ''} --kid --isc --input1 {str(sample_dir)} --input2 {str(og_dir)} --kid-subset-size {n_kid}"

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
		# fid = float(
		#     re.search('(?<=frechet_inception_distance:).*(?=\n)',
		#               output).group(0))
		kid = float(
			re.search('(?<=kernel_inception_distance_mean:).*(?=\n)',
					  output).group(0))
		kid_std = float(
			re.search('(?<=kernel_inception_distance_std:).*(?=\n)',
					  output).group(0))
		result_dict= {'IS': is_mean,'IS_std': is_std, 'KID': kid, 'KID_std' : kid_std}
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
				   work_dir / f"Sample output {counter}.jpg",
				   normalize=True)
		counter += 1
	return sample_img_list


def compute_lpips(data_fpath, sample_img_list):
	LPIPS_score_list = []
	for i, img1 in enumerate(sample_img_list):
		for img2 in sample_img_list[i+1:]:
			LPIPS_score_list.append(
				loss_fn_alex(img1, img2).squeeze().item())

	avg = np.mean(LPIPS_score_list)
	stderr = np.std(np.array(LPIPS_score_list)) / np.sqrt(
		len(LPIPS_score_list))
	print(f'LPIPS: {avg}')
	return {'LPIPS': avg, 'LPIPS_std': stderr}

def get_metrics_for_run(sample_img_noise,generator_fpath, data_fpath, work_dir, n_gpu = 0 ):
	sample_img_list = generate_samples(sample_img_noise, generator_fpath)
	lpips_dict = compute_lpips(data_fpath, sample_img_list)
	fid_kid_is_dict = get_fid_kid(data_fpath, work_dir, n_gpu)
	return {**lpips_dict, **fid_kid_is_dict}

def save_image_grid(sample_img_noise, generator_fpath, title, idxs = [0,1,2,3]):
	netG = Generator(n_gpu).to(device)

	netG.load_state_dict(
		torch.load(generator_fpath, map_location=torch.device('cpu')))

	sample_img_output = netG(sample_img_noise).to('cpu')

	img_out = np.transpose(
		vutils.make_grid(sample_img_output[np.ix_(idxs)], padding=2, normalize=True),
		(1, 2, 0))
	plt.imshow(img_out)
	plt.axis('off')
	plt.savefig(f"{results_dir}/{title}", bbox_inches='tight', transparent=True, pad_inches=0)


if __name__ == '__main__':
	loss_fn_alex = lpips.LPIPS(net='alex')

	work_dir = plib.Path.cwd()/"sample_working_dir"
	work_dir.mkdir(exist_ok=True)
	results_dir = plib.Path.cwd()/"results"
	results_dir.mkdir(exist_ok=True)

	device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	n_gpu = torch.cuda.device_count()

	manualSeed = 999
	torch.manual_seed(manualSeed)

	sample_img_noise = torch.randn(N_SAMPLES,N_SAMPLES,1,1,device=device)
	grid_fixed_noise_idxs = [2, 61, 21]

	pretrained_g = 'celeba_pretrained_generator'

	#### dataset experiment ###
	experiment_name = 'dataset'
	print(experiment_name)
	experiment_saved_model_dir = f'saved_model/{experiment_name}_saved_models'
	generator_fpaths = sorted(glob.glob(f"{experiment_saved_model_dir}/*netG.pt"))
	datasets = ['CelebA_Bald','CelebA_Eyeglasses','CelebA_Bangs', '100-shot-obama', 'AnimalFace-cat']
	i = 0
	result_dict = defaultdict(dict)
	for dataset_name in datasets:
		for reg in ['EWC', 'No EWC']:
			print(dataset_name, reg)
			generator_fpath = generator_fpaths[i]
			data_fpath = f'data/{dataset_name}_size_64'

			result_dict[dataset_name][reg] = get_metrics_for_run(sample_img_noise,
				generator_fpath, data_fpath, work_dir)

			save_image_grid(sample_img_noise,
				generator_fpath,
				title = f'dataset_{dataset_name}_{reg}.jpg',
				idxs = grid_fixed_noise_idxs)
			i+=1

	outpath = f'results/{experiment_name}_experiment_results.json'
	with open(outpath, 'w') as f:
		json.dump(result_dict, f, indent=4)

	#### n-shot experiment ###
	experiment_name = 'n_shot'
	print(experiment_name)
	experiment_saved_model_dir = f'saved_model/{experiment_name}_saved_models'
	generator_fpaths = sorted(glob.glob(f"{experiment_saved_model_dir}/*netG.pt"))

	dataset_name = 'CelebA_Eyeglasses'
	n_shot_runs = [100, 50, 10, 3]

	i = 0
	result_dict = defaultdict(dict)
	for n in n_shot_runs:
		for reg in ['EWC', 'No EWC']:
			print(n, reg)
			generator_fpath = generator_fpaths[i]
			if n == 100:
				data_fpath = f'data/{dataset_name}_size_64'
			else:
				data_fpath = f'log/few_shot_datasets/run_10_{n}_shots'

			result_dict[n][reg] = get_metrics_for_run(sample_img_noise,
				generator_fpath, data_fpath, work_dir)
			save_image_grid(sample_img_noise,
				generator_fpath,
				title = f'nshot_{n}_{reg}.jpg',
				idxs = [36])
			i+=1

	save_image_grid(sample_img_noise,
		pretrained_g,
		title = f'n_shot_pretrained.jpg',
		idxs = [36])

	outpath = f'results/{experiment_name}_experiment_results.json'
	with open(outpath, 'w') as f:
		json.dump(result_dict, f, indent=4)

	#### Discriminator Handicap Experiment ###
	experiment_name = 'method_compare'
	print(experiment_name)
	experiment_saved_model_dir = f'saved_model/{experiment_name}_saved_models'
	generator_fpaths = sorted(glob.glob(f"{experiment_saved_model_dir}/*netG.pt"))

	dataset_name = 'CelebA_Bald'
	method_runs = ['LS', 'IN', 'LS+IN', 'D_EWC']
	i = 0
	result_dict = defaultdict(dict)
	for m in method_runs:
		for reg in ['EWC', 'No EWC']:
			print(m, reg)
			generator_fpath = generator_fpaths[i]
			data_fpath = f'data/{dataset_name}_size_64'

			result_dict[m][reg] = get_metrics_for_run(sample_img_noise,
				generator_fpath, data_fpath, work_dir)
			save_image_grid(sample_img_noise,
				generator_fpath,
				title = f'method_{m}_{reg}.jpg',
				idxs = [59])
			i+=1

	save_image_grid(sample_img_noise,
		pretrained_g,
		title = f'method_pretrained.jpg',
		idxs = [59])

	outpath = f'results/{experiment_name}_experiment_results.json'
	with open(outpath, 'w') as f:
		json.dump(result_dict, f, indent=4)
