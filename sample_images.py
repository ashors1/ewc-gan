import argparse
import os
import lpips
import numpy as np
from tqdm import tqdm
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torchvision.utils as vutils
import matplotlib.pyplot  as plt
from dcgan import Generator, Discriminator
from tqdm import  tqdm


'''
Samples images from trained Generator

'''

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pre_G', type=str)
parser.add_argument('-o', '--out_dirname', type=str, default = '')
parser.add_argument('-N', type=int, default=100)
parser.add_argument('--sample_grid_fname', type=str, default='')

opt = parser.parse_args()

if __name__ == '__main__':

	if opt.out_dirname != '':
		outpath = f'sampled_images/{opt.out_dirname}'
		if not os.path.exists(outpath):
			os.makedirs(outpath)

	device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	############################################
	###### Model Initialization
	############################################

	### map_location for running on cpu
	n_gpu = torch.cuda.device_count()
	netG = Generator(n_gpu).to(device)
	netG.load_state_dict(
		torch.load(opt.pre_G, map_location=torch.device('cpu')))


	############################################
	###### Sample Images
	############################################

	nz = 100
	manualSeed = 999
	torch.manual_seed(manualSeed)

	#write out individual samples for running metrics on
	noise = torch.randn(opt.N, nz, 1, 1, device=device)
	# Generate fake image batch with G
	fake = netG(noise)

	if opt.out_dirname != '':
		for i in tqdm(range(opt.N), desc = f'Writing Generated Images to {outpath}'):
			img_grid = np.transpose(
				vutils.make_grid(fake[i], padding=0, normalize=True),
				(1, 2, 0))
			plt.imshow(img_grid)
			plt.axis('off')
			plt.savefig(f"{outpath}/{i}.jpg", bbox_inches='tight', transparent=True, pad_inches=0)

	#write out a sample grid to the result folder for paper
	if opt.sample_grid_fname != '':
		results_dir = './results'

		if not os.path.exists(results_dir):
			os.makedirs(results_dir)

		img_out = np.transpose(
			vutils.make_grid(fake[0:4], padding=2, normalize=True),
			(1, 2, 0))
		plt.imshow(img_out)
		plt.axis('off')
		plt.savefig(f"{results_dir}/{opt.sample_grid_fname}", bbox_inches='tight', transparent=True, pad_inches=.1)
