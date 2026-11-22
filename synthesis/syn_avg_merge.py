import os

import torch

from config import args
from common_utils import get_diff_model, get_weights, weight_aggregate
from data.data_loder import get_fl_train_dataloader
from synthesis.syn_utils import ddpm_sample, ddim_sample

if __name__ == "__main__":
    img_path = f"{args.img_path}/avg_merge"
    os.makedirs(img_path, exist_ok=True)

    model = get_diff_model()

    clients = list(range(args.num_clients))
    client_loaders, _, _ = get_fl_train_dataloader()

    weights = get_weights(clients, client_loaders)
    # weights = [1 / args.num_clients for _ in clients]

    global_state = None
    for i, client in enumerate(clients):
        client_state = torch.load(f"{args.model_path}/client{client}.pth", )
        global_state = weight_aggregate(global_state, client_state, weights[i])

    model.load_state_dict(global_state)

    if args.ddim_sample:
        ddim_sample(model, img_path, show_tqdm=True)
    else:
        ddpm_sample(model, img_path, show_tqdm=True)
