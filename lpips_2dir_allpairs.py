import argparse
import os
import lpips
import numpy as np
from tqdm import  tqdm

'''
Script Adapted from PerceptualSimilarity Repo
<https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_1dir_allpairs.py>
to compute all pairwise LPIPS distances between images in two directories

Example Usage
-------------

> python lpips_2dir_allpairs.py -d0 data/100-shot-obama/ -d1 data/100-shot-grumpy_cat -o obama_grumpycat_pairwise.txt



'''

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('-N', type=int, default=None)
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories

f = open(opt.out,'w')
files_0 = os.listdir(opt.dir0)
files_1 = os.listdir(opt.dir1)
if(opt.N is not None):
	files_0 = files_0[:opt.N]
	files_1 = files_1[:opt.N]
F = len(files_0)

dists = []
for (ff,file) in enumerate(tqdm(files_0, desc = 'Computing pairwise LPIPS distances...')):
	img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
	if(opt.use_gpu):
		img0 = img0.cuda()

	for file1 in files_1:
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file1)))

		if(opt.use_gpu):
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		# print('(%s,%s): %.3f'%(file,file1,dist01))
		f.writelines('(%s,%s): %.6f\n'%(file,file1,dist01))

		dists.append(dist01.item())


avg_dist = np.mean(np.array(dists))
stderr_dist = np.std(np.array(dists))/np.sqrt(len(dists))

print('Avg: %.5f +/- %.5f'%(avg_dist,stderr_dist))
f.writelines('Avg: %.6f +/- %.6f'%(avg_dist,stderr_dist))

f.close()
