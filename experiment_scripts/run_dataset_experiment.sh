## Experiment 2: Comparing datasets of varying domain distances with and without EWC

# TODO: we can either run both EWC & No EWC with the best hyperparam/method combo justified by Experiment 1
# or run both with defaults (currently, the calls below are just with the defaults )

# TODO: decide on 3/4 of the datasets: Bald, Eyeglasses, Bangs, Mustache and 2 other non-subsets:
# A far domain (artist? -- simpsons was terrible) and another face domain
# (could do obama, but the closer LPIPS distance for obama than some of the subsets may be a bit confusing )

# The following actually should have already been run in deciding which subset to use in experiment 1, but
# need to run this for the first time for the two non-CelebA datasets
python3 train.py --data_root data/CelebA_Bald_size_64/ --batch_size 4 --save --score_freq -1 --num_epochs 100
python3 train.py --data_root data/CelebA_Bald_size_64/ --batch_size 4 --save --score_freq -1 --num_epochs 100 --G_ewc_lambda 0

python3 train.py --data_root data/CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq -1 --num_epochs 6
python3 train.py --data_root data/CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq -1 --num_epochs 6 --G_ewc_lambda 0

python3 train.py --data_root data/CelebA_Bangs_size_64/ --batch_size 4 --save --score_freq -1 --num_epochs 10
python3 train.py --data_root data/CelebA_Bangs_size_64/ --batch_size 4 --save --score_freq -1 --num_epochs 10 --G_ewc_lambda 0

python3 train.py --data_root data/CelebA_Mustache_size_64/ --batch_size 4 --save --score_freq -1 --num_epochs 15
python3 train.py --data_root data/CelebA_Mustache_size_64/ --batch_size 4 --save --score_freq -1 --num_epochs 15 --G_ewc_lambda 0
