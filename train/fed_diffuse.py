import math

import numpy as np
import torch
from tqdm import tqdm

from common_utils import get_weights, logger, get_diff_model, get_train_scheduler
from config import args
from data.data_loder import get_fl_train_dataloader
from train.train_utils import get_optimizer, weight_aggregate, get_warmup_steps

if __name__ == '__main__':
    clients = np.arange(args.num_clients)
    client_loaders = get_fl_train_dataloader()

    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    num_epochs_per_round = math.ceil(args.client_epochs / args.rounds)

    global_model = get_diff_model()

    train_scheduler = get_train_scheduler()

    check_path = args.check_path / "fed_diffuse"
    check_path.mkdir(parents=True, exist_ok=True)
    logger.add_log_path(check_path)

    for cur_round in tqdm(range(1, args.rounds + 1), dynamic_ncols=True):
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
            warmup_steps = get_warmup_steps(client_loader)
            avg_loss = 0
            for epoch in range(0, num_epochs_per_round):
                update_step = ((cur_round - 1) * num_epochs_per_round + epoch) * len(client_loader)
                total_loss = 0.0
                for x0, *_ in client_loader:
                    x0 = x0.to(args.device)
                    B = x0.shape[0]
                    t = torch.randint(0, args.T, (B,), device=args.device).long()
                    noise = torch.randn_like(x0)
                    xt = train_scheduler.add_noise(x0, noise, t)
                    noise_pred = local_model(xt, t).sample
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    update_step += 1
                    lr = args.lr * min(update_step / warmup_steps, 1.0)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(client_loader)
            logger.info(f"Round{cur_round} - Rank {i} - Client {client} Loss = {avg_loss:.4f}")
            client_state = local_model.state_dict()
            global_state = weight_aggregate(global_state, client_state, weights[i])

            del local_model
            del optimizer

        torch.save({
            "round": cur_round,
            "model": global_state,
        }, check_path / f"{cur_round}.pth")

        global_model.load_state_dict(global_state)
