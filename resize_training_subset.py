import argparse
import os
from tqdm import tqdm
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torchvision.utils as vutils
import matplotlib.pyplot  as plt
import numpy as np

'''
Sample N training images, resize, and write each image to file
'''

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	description="Sample N training images, resize, and write each image to file")
parser.add_argument('-d_in','--dir_input', type=str, default='./data/CelebA/')
parser.add_argument('-d_out','--dir_output', type=str, default='./data/CelebA_sample_resized/')
parser.add_argument('-s','--size', type=int, default=64)
parser.add_argument('-N', type=int, default=100)
parser.add_argument('--sample_grid_fname', type=str, default='')

opt = parser.parse_args()

if __name__ == '__main__':

	if not os.path.exists(opt.dir_output):
		os.makedirs(opt.dir_output)

	pretrained_dataset = dset.ImageFolder(
	    root=opt.dir_input,
	    transform=transforms.Compose([
	        transforms.Resize(opt.size),
	        transforms.CenterCrop(opt.size),
	        transforms.ToTensor(),
	        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	    ]))
	# Create the dataloader
	dataloader = torch.utils.data.DataLoader(
	    pretrained_dataset,
	    batch_size=opt.N,
	    shuffle=True,
	    num_workers=1)

	data_0 = next(iter(dataloader))

	for i, img in enumerate(tqdm(data_0[0], desc = f'Writing resized images to {opt.dir_output}')):

		img_out = np.transpose(
			vutils.make_grid(img, padding=2, normalize=True),
			(1, 2, 0))
		plt.imshow(img_out)
		plt.axis('off')
		plt.savefig(f"{opt.dir_output}/{i}.jpg", bbox_inches='tight', transparent=True, pad_inches=0)


	#write out a sample grid to the result folder for paper
	if opt.sample_grid_fname != '':
		results_dir = './results'

		if not os.path.exists(results_dir):
			os.makedirs(results_dir)

		img_out = np.transpose(
			vutils.make_grid(data_0[0][0:4], padding=2, normalize=True),
			(1, 2, 0))
		plt.imshow(img_out)
		plt.axis('off')
		plt.savefig(f"{results_dir}/{opt.sample_grid_fname}", bbox_inches='tight', transparent=True, pad_inches=.1)
