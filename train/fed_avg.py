import math
import os

import numpy as np
import torch
from tqdm import tqdm

from common_utils import get_weights, logger, get_diff_model, get_train_scheduler
from config import args
from data.data_loder import get_fl_train_dataloader
from train.train_utils import train_epoch, get_optimizer, weight_aggregate

if __name__ == '__main__':
    clients = np.arange(args.num_clients)
    client_loaders = get_fl_train_dataloader()

    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    num_epochs_per_round = math.ceil(args.client_epochs / args.rounds)

    global_model = get_diff_model()

    train_scheduler = get_train_scheduler()

    method = "fed_avg"
    check_path = f"{args.check_path}/{method}"
    os.makedirs(check_path, exist_ok=True)
    logger.add_log_path(check_path)

    start_round = 1
    for cur_round in tqdm(range(start_round, args.rounds + 1), dynamic_ncols=True):
        logger.info(f"Start round {cur_round}-------------------")
        selected_clients = np.random.choice(
            clients, num_clients_per_round, replace=False
        )
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            local_model = get_diff_model()
            local_model.load_state_dict(global_model.state_dict())
            optimizer = get_optimizer(local_model)
            client_loader = client_loaders[client]
            loss = -1
            for epoch in range(0, num_epochs_per_round):
                update_steps = (cur_round * num_epochs_per_round + epoch) * len(client_loader)
                loss, _ = train_epoch(local_model, optimizer, client_loader,
                                      train_scheduler, update_steps)
            logger.info(f"Rank {i} - Client {client} Loss = {loss:.4f}")
            client_state = local_model.state_dict()
            global_state = weight_aggregate(global_state, client_state, weights[i])

            del local_model
            del optimizer

        torch.save({
            "round": cur_round,
            "model": global_state,
        }, f"{check_path}/fed_avg_{cur_round}.pth")

        global_model.load_state_dict(global_state)
