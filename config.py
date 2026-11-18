import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch.utils.data
import yaml
from diffusers import UNet2DModel, DDPMScheduler
from torch.utils.data import DataLoader

from data.data_loder import partition_client_data, PrunableDataset
from data.fetch import fetch_dataset_cifar10


def parse_args():
    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser()

    parser.add_argument("--server", choices=config["servers"])

    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--record_data", type=bool)
    parser.add_argument("--device")
    parser.add_argument("--test_batchsize", type=int)
    parser.add_argument("--num_clients", type=int)
    parser.add_argument("--dirichlet_alpha", type=float)
    parser.add_argument("--client_id", type=int)
    parser.add_argument("--num_samplings", type=int)

    parser.set_defaults(**config["run"])

    args = parser.parse_args([] if "ipykernel" in sys.modules else None)

    vars(args).update(config["training"])
    vars(args).update(config["servers"][args.server])
    vars(args).update(config["datasets"][args.dataset])

    """
       1. results, given by args.results_path
       2. seed_clients__dir, optional, ignored now
       3. dataset, given by args.dataset

       - models
         - os
       - imgs
         - ws
         - cl
         - dfd
   """
    args.results_path = f"{args.results_path}/{args.dataset}"

    return args


args = parse_args()


def init_training():
    """
    init for model training
    """

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

        linear_scheduler = DDPMScheduler(
            num_train_timesteps=args.T,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"
        )

    client_loaders, validate_loader = partition_client_data(
        train_data, batch_size=args.batch_size
    )

    test_loader = DataLoader(
        PrunableDataset(test_data, range(len(test_data))),
        batch_size=args.test_batchsize,
        shuffle=False,
    )

    global_model.to(args.device)

    return test_loader, client_loaders, validate_loader, global_model, linear_scheduler


test_loader, client_loaders, validate_loader, global_model, linear_scheduler = init_training()

logger = logging.getLogger("ws")
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(handler)
