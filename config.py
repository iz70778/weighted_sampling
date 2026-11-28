import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch.utils.data
import yaml


def parse_args():
    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--record_data", type=bool)
    parser.add_argument("--test_batchsize", type=int)
    parser.add_argument("--num_clients", type=int)
    parser.add_argument("--dirichlet_alpha", type=float)
    parser.add_argument("--num_samplings", type=int)
    parser.add_argument("--ddim", type=bool)

    # parser.add_argument("--rounds", type=int)

    parser.set_defaults(**config["run"])

    args = parser.parse_args([] if "ipykernel" in sys.modules else None)

    vars(args).update(config["training"])
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
    args.results_path = Path(args.results_path) / args.dataset
    args.img_path = args.results_path / "imgs"
    args.model_path = args.results_path / "models"
    args.check_path = args.model_path / "checkpoints"

    args.img_path.mkdir(parents=True, exist_ok=True)
    args.check_path.mkdir(parents=True, exist_ok=True)

    return args


args = parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


