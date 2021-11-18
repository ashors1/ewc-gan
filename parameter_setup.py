import torch
import argparse


def parameter_setup():

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument("-p",
                        "--pretrain",
                        help="Turn on training model from scratch",
                        action="store_true",
                        default=False)
    parser.add_argument("-s",
                        "--save",
                        help="Turn on saving the generator and discriminator",
                        action="store_true",
                        default=False)

    parser.add_argument("--pre_G",
                        help="setting location for pre-trained Generator",
                        #default="./netG_10_epoch_state_dict")
                        default='../../celeba_pretrained_generator')
    parser.add_argument("--pre_D",
                        help="setting location for pre-trained Discriminator",
                        #default="./netD_10_epoch_state_dict")
                        default='../../celeba_pretrained_discriminator')
    parser.add_argument("--data_root",
                        help="setting location for training data",
                        #default="./data/AF_Mini")
                        default="../../few_shot")

    parser.add_argument("--batch_size",
                        type=int,
                        help="setting batch_size",
                        default=4)

    parser.add_argument(
        "--img_freq",
        type=int,
        help="setting frequency (every n iteration) of saving images",
        default=50)

    parser.add_argument("--image_size",
                        type=int,
                        help="setting image_size",
                        default=64)
    parser.add_argument("--workers",
                        type=int,
                        help="setting workers for data load",
                        default=2)
    parser.add_argument("--num_epochs",
                        type=int,
                        help="setting number of epochs",
                        default=100)

    parser.add_argument("--D_lr",
                        type=float,
                        help="Setting learning rate for discriminator",
                        #default=0.0002)
                        default=0.0006)

    parser.add_argument("--D_update_rate",
                        type=int,
                        help="setting the discriminator update rate",
                        default=1)

    parser.add_argument(
        "--D_beta1",
        type=float,
        help="Setting learning rate for discriminator Adam optimizer beta 1",
        default=0.5)
    parser.add_argument(
        "--D_beta2",
        type=float,
        help="Setting learning rate for discriminator Adam optimizer beta 2",
        default=0.999)
    parser.add_argument("--G_lr",
                        type=float,
                        help="Setting learning rate for generator",
                        default=0.0002)
    parser.add_argument(
        "--G_beta1",
        type=float,
        help="Setting learning rate for generator Adam optimizer beta 1",
        default=0.5)
    parser.add_argument(
        "--G_beta2",
        type=float,
        help="Setting learning rate for generator Adam optimizer beta 2",
        default=0.999)

    parser.add_argument("--ngpu",
                        type=int,
                        help="Number of GPU available",
                        default=torch.cuda.device_count())

    # EWC Parameters
    parser.add_argument("--ewc_data_root",
                        help="setting location for ewc evaluation data",
                        #default="./data/AF_Mini")
                        default="../../few_shot")

    parser.add_argument("--ewc_lambda",
                        type=float,
                        help="Setting ewc penalty lambda coefficient ",
                        default=10000)

    args = parser.parse_args()
    train_dict = dict()

    for ele in args._get_kwargs():
        train_dict[ele[0]] = ele[1]

    train_dict["device"] = torch.device("cuda:0" if (
        torch.cuda.is_available() and train_dict['ngpu'] > 0) else "cpu")

    # training setup for EWC
    ewc_dict = {
        "ewc_lambda": train_dict['ewc_lambda'],
        "ewc_data_root": train_dict['ewc_data_root']
    }

    return train_dict, ewc_dict
