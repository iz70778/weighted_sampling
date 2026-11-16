import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    '''specify'''
    server_prefix = "/scratch-calder/users/sc21s2j"
    parser.add_argument("--result_fp", type=str, default=f"{server_prefix}/ws/results/")
    parser.add_argument("--dataset_dir", type=str, default=f"{server_prefix}/dataset/")
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

    res.batch_size = 32
    # res.test_batchsize = 32
    res.train_ratio = 1.0  # for methods requiring validation
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