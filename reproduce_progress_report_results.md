## Reproduce progress report results

The following commands can be be used to reproduce the figures/metrics for the progress report.

Assumes the following structure for the `data` folder:
```
data
├── 100-shot-obama
│   └── 0
│       └── *.jpg
└── CelebA
    └── img_align_celeba
│       └── *.jpg
```

#### Fine-tune models

This assumes that the model has already been pre-trained on CelebA (64x64) and finetuned on the 100-shot obama set (64x64) with and without EWC by running the following, as Run 0 and Run 1 respectively:

```
python3 train.py --data_root data/100-shot-obama --save --G_lr=.0001 --ewc_lambda 100000 --num_epochs 770 --batch_size 32 # Run 0
python3 train.py --data_root data/100-shot-obama --save --G_lr=.0001 --ewc_lambda 0 --num_epochs 770 --batch_size 32 #Run 1
```

#### Save resized/cropped images

Write Resize/cropped (64x64) few shot and 100 samples of pretraining set, and write a sample 1x4 grid of the first 4 images for report:
```
python3 resize_training_subset.py -d_in data/100-shot-obama/ -s 64 -d_out data/100-shot-obama_size_64 --sample_grid_fname obama_grid.jpg
python3 resize_training_subset.py -d_in data/CelebA -d_out data/CelebA_samples_size_64  -s 64 -N 100 --sample_grid_fname celeba_grid.jpg
```

#### General model samples

Write 100 generated samples from pre-trained and each fine-tuned model, and write a sample 1x4 grid of the first 4 images images for report:
```
python3 sample_images.py --pre_G netG_10_epoch_state_dict --o generated_celebA --sample_grid_fname generated_celebA
python3 sample_images.py --pre_G saved_model/Run_0_netG.pt --o generated_obama --sample_grid_fname generated_obama
python3 sample_images.py --pre_G saved_model/Run_1_netG.pt --o generated_obama_no_ewc --sample_grid_fname generated_obama_no_ewc
```
Note that the `log/Run {i} Fixed Noise Output at Iter -1.png` produced at the end of fine-tuning can also be used as the 1x4 sample grid for the fine-tuned models.

#### Get metrics & domain distances

To get FID for each model, first install package:

```
pip install pytorch-fid
```

then run:

```
python3 -m pytorch_fid  data/100-shot-obama_size_64/ sampled_images/generated_obama/ --num-workers 1
python3 -m pytorch_fid  data/100-shot-obama_size_64/ sampled_images/generated_obama_no_ewc/ --num-workers 1
```

To get domain distances and generated image diversity, we use the LPIPS metric. In order to use command line scripts, clone the LPIPS repo
in addition to installing the package:

```
pip install lpips
git clone git@github.com:richzhang/PerceptualSimilarity.git
```

Get domain distance between resized pre-train and few shot set (takes a while so run with smaller `-N` to test):

```
mkdir distances
python lpips_2dir_allpairs.py -d0 data/CelebA_samples_size_64/ -d1 data/100-shot-obama_size_64/ -o distances/celebA_obama.txt
```

Get generated image diversity of sampled images:

```
python ../PerceptualSimilarity/lpips_1dir_allpairs.py -d sampled_images/generated_obama/ -o distances/obama_generated.txt
python ../PerceptualSimilarity/lpips_1dir_allpairs.py -d sampled_images/generated_obama_no_ewc/ -o distances/obama_generated_no_ewc.txt
```

#### Produce plot of EWC loss

Produce EWC loss plot:
```
python3 plot_ewc_loss.py
```

Note that that script can be easily modified to produce other plots comparing logged info across runs
