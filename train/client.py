import math
import copy
import torch
from torch import nn
from config import args
from train.diff_utils import linear_scheduler

import torch.nn.functional as F


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


def train_diff_os(client, client_loader, global_model,):
    client_model = copy_global_model(global_model)
    optimizer = torch.optim.AdamW(
        client_model.parameters(),
        lr=args.lr
    )
    for epoch in range(args.client_epochs):
        client_model.train()
        total_loss = 0.0
        # progress = tqdm(client_loader, desc=f"Epoch {epoch+1}/{args.client_epochs}")
        for x0, _, _, _ in  client_loader:
            x0 = x0.to(args.device)
            B = x0.shape[0]
            t = torch.randint(
                0, args.T, (B,), device=args.device
            ).long()
            noise = torch.randn_like(x0)
            xt = linear_scheduler.add_noise(x0, noise, t)
            noise_pred = client_model(xt, t).sample
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(client_loader)
        print(f"Client {client} - Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Save model after every epoch
        torch.save(
            client_model.state_dict(),
             f"{args.results_path}/client{client}_ddpm_model.pth"
        )

    del client_model
    del optimizer

