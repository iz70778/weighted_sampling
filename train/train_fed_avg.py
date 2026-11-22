import math

import numpy as np
from tqdm import tqdm

from common_utils import get_weights, weight_aggregate
from config import args


def fed_avg(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        selected_clients = np.random.choice(
            clients, num_clients_per_round, replace=False
        )
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])
        global_model.load_state_dict(global_state)
        round_accuracy.append(0)
    return round_accuracy


