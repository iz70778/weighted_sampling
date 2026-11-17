import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    '''specify'''
    parser.add_argument("--server", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--option", type=str, default="")
    parser.add_argument("--record_data", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_batchsize", type=int, default=32)

    """server"""
    parser.add_argument("--num_clients", type=int)

    """client"""
    parser.add_argument("--client_id", type=int)

    """dataset"""
    parser.add_argument("--dirichlet_alpha", type=float)
    parser.add_argument("--dataset", type=str)

    if "ipykernel" in sys.modules:
        res = parser.parse_args([])
    else:
        res = parser.parse_args()

    if res.server == "calder":
        server_prefix = "/scratch-calder/users/sc21s2j"
        res.results_dir = f"{server_prefix}/weighted_sampling/results/"
        res.dataset_dir = f"{server_prefix}/dataset/"
    elif res.server == "aire":
        server_prefix = "/scratch/users/sc21s2j"
        res.results_dir = f"{server_prefix}/weighted_sampling/results/"
        res.dataset_dir = f"{server_prefix}/dataset/"
    elif res.server == "swjtu":
        server_prefix = "/home/sc21s2j"
        res.results_dir = f"{server_prefix}/weighted_sampling/results/"
        res.dataset_dir = f"{server_prefix}/dataset/"
    elif res.server == "mine":
        server_prefix = "/users/sc21s2j"
        res.results_dir = f"{server_prefix}/weighted_sampling/results/"
        res.dataset_dir = f"{server_prefix}/dataset/"
    else:
        raise ValueError(f"Unknown server: {res.server}")


    res.batch_size = 32
    res.train_ratio = 1.0
    res.client_selected_ratio = 0.15

    res.num_classes = 10
    res.rounds = 100
    res.pre_train_round = 10

    if  res.dataset in ['cifar10']:
        res.client_epochs = 50
        res.image_size = 32
        res.batch_size = 32
        res.lr = 1e-4
        res.T = 1000
    return res

args = parse_args()