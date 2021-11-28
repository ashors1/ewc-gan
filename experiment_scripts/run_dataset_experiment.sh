## Experiment 2: Comparing datasets of varying domain distances with and without EWC

# TODO: we can either run both EWC & No EWC with the best hyperparam/method combo justified by Experiment 1
# or run both with defaults (currently, the calls below are just with the defaults )

# TODO: decide on 3/4 of the datasets: Bald, Eyeglasses, Bangs, Mustache and 2 other non-subsets:
# A far domain (artist? -- simpsons was terrible) and another face domain
# (could do obama, but the closer LPIPS distance for obama than some of the subsets may be a bit confusing )

# The following actually should have already been run in deciding which subset to use in experiment 1, but
# need to run this for the first time for the two non-CelebA datasets
python3 train.py --data_root ../../CelebA_Bald_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 60 --G_lr 0.001
python3 train.py --data_root ../../CelebA_Bald_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 60 --G_ewc_lambda 0 --G_lr 0.001

python3 train.py --data_root ../../CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 6 --G_lr 0.001
python3 train.py --data_root ../../CelebA_Eyeglasses_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 6 --G_ewc_lambda 0 --G_lr 0.001

python3 train.py --data_root ../../CelebA_Bangs_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 10 --G_lr 0.001
python3 train.py --data_root ../../CelebA_Bangs_size_64/ --batch_size 4 --save --score_freq 0 --num_epochs 10 --G_ewc_lambda 0 --G_lr 0.001

python3 train.py --data_root ../../few_shot --batch_size 4 --save --score_freq 0 --num_epochs 34 --G_ewc_lambda 1500 --G_lr 0.001
python3 train.py --data_root ../../few_shot --batch_size 4 --save --score_freq 0 --num_epochs 34 --G_ewc_lambda 0 --G_lr 0.001

python3 train.py --data_root ../../100_shot_cat --batch_size 4 --save --score_freq 0 --num_epochs 34 --G_ewc_lambda 1000 --G_lr 0.001
python3 train.py --data_root ../../100_shot_cat --batch_size 4 --save --score_freq 0 --num_epochs 34 --G_ewc_lambda 0 --G_lr 0.001
