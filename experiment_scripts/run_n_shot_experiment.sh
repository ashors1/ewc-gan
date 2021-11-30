## Experiment 1: Handicapping the discriminator (Qualitative Results only)

## TODO: After some experimentation, seems like these are the best number of epochs to run for these subsets
## Based on these results, need to decide which dataset to use for demonstrating the method comparison

# data/CelebA_Eyeglasses_size_64/ : --num_epochs 6 (Results in `logs_experiments.zip`)
# data/CelebA_Bald_size_64/ : --num_epochs 100 (Results in `logs_experiments.zip`) (we could maybe try running this for fewer)
# data/CelebA_Mustache_size_64/ : --num_epochs 15 (TODO)
# data/CelebA_Bangs_size_64/ : --num_epochs 10 (TODO)

python3 train.py --data_root ../../CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 6 --G_lr 0.001
python3 train.py --data_root ../../CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 6 --G_lr 0.001 --G_ewc_lambda 0

python3 train.py --data_root ../../CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 12 --G_lr 0.001 --num_shots 50 --G_ewc_lambda 1000
python3 train.py --data_root ../../CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 12 --G_lr 0.001 --num_shots 50 --G_ewc_lambda 0

python3 train.py --data_root ../../CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 60 --G_lr 0.001 --num_shots 10 --G_ewc_lambda 2000
python3 train.py --data_root ../../CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 60 --G_lr 0.001 --num_shots 10 --G_ewc_lambda 0

python3 train.py --data_root ../../CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 200 --G_lr 0.001 --num_shots 3 --G_ewc_lambda 3000
python3 train.py --data_root ../../CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 200 --G_lr 0.001 --num_shots 3 --G_ewc_lambda 0

