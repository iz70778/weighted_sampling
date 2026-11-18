import copy
import math

import torch
from torch import nn

from config import args

criterion = nn.CrossEntropyLoss()


def copy_global_model(global_model):
    global_model_copy = copy.deepcopy(global_model)
    global_model_copy.to(args.device)
    return global_model_copy

def get_lr(current_round):
    if args.dataset in ['mnist', 'cifar10', 'fmnist', 'cinic10']:
        return pow(0.99, current_round - 1) * args.lr
    elif args.dataset in ['cifar100', 'tiny_imagenet']:
        cos_inner = math.pi * current_round / args.rounds
        return 0.5 * args.lr * (1 + math.cos(cos_inner))
    else:
        raise 'lr decay strategy is not set'


def train_client(client_loader, global_model, cur_round):
    # global model is on gpu
    client_model = copy_global_model(global_model)
    client_model.train()
    lr = get_lr(cur_round)
    optimizer = torch.optim.SGD(client_model.parameters(), lr)
    total = 0
    for epoch in range(args.client_epochs):
        for x, y, _, _ in client_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            out = client_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += x.shape[0]
    local_state = client_model.state_dict()

    # free cache
    del client_model
    del optimizer

    return local_state


