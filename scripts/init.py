import random

from torch.utils.data import DataLoader

from config import args
from train.model import CNN_MNIST, CNN_CIFAR_10, CNN_FMNIST, ResNet_CIFAR100, ResNet_18
from diffusers import UNet2DModel

random.seed(args.seed)

import numpy as np

np.random.seed(args.seed)

import torch.utils.data

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from data.data_loder import partition_client_data, PrunableDataset
from data.fetch import (
    fetch_dataset_mnist,
    fetch_dataset_fmnist,
    fetch_dataset_cifar10,
    fetch_dataset_cifar100,
    fetch_dataset_tiny_imagenet,
    fetch_dataset_cinic10,
)
from config import args
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)


def init_training():
    """
    init for model training
    """

    if args.dataset == "cifar10":
        train_data, test_data = fetch_dataset_cifar10()
        global_model = UNet2DModel(
            sample_size=args.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            ),
            norm_num_groups=32,
        ).to(args.device)

    
    client_loaders, validate_loader = partition_client_data(
        train_data, batch_size=args.batch_size
    )

    test_loader = DataLoader(
        PrunableDataset(test_data, range(len(test_data))),
        batch_size=args.test_batchsize,
        shuffle=False,
    )

    global_model.to(args.device)

    return test_loader, client_loaders, validate_loader, global_model


test_loader, client_loaders, validate_loader, global_model = init_training()
