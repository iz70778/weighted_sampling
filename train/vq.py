import math

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

from config import args
from data.data_loder import get_fl_train_dataloader
from common_utils import get_weights, logger, get_diff_model, get_train_scheduler
from config import args
from data.data_loder import get_fl_train_dataloader
from train.train_utils import get_optimizer, weight_aggregate, get_warmup_steps, save_ema_model


def prepare_codebooks():
    save_path = args.check_path / "codebooks"
    save_path.mkdir(parents=True, exist_ok=True)

    weights = Inception_V3_Weights.IMAGENET1K_V1
    encoder = inception_v3(weights=weights)
    encoder.fc = nn.Identity()
    encoder = encoder.eval().cuda()
    preprocess = weights.transforms()

    client_loaders = get_fl_train_dataloader()
    print("Start constructing codebooks")
    with torch.no_grad():
        for client, client_loader in tqdm(enumerate(client_loaders), total=len(client_loaders)):
            features = []
            idxs = []
            for x, _, gidx, _ in client_loaders[client]:
                x = (x.cuda() + 1.0) / 2.0  # undo transforms
                features.append(encoder(preprocess(x.cuda())).float().cpu())
                idxs.extend(gidx.tolist())
            features = torch.cat(features)

            kmeans = KMeans(
                n_clusters=256,
                init="k-means++",
                n_init=10,
                max_iter=50,
                algorithm="lloyd",
                random_state=args.seed + client,
            ).fit(features.numpy())

            torch.save({
                "cen": torch.from_numpy(kmeans.cluster_centers_),
                "idxs": dict(zip(idxs, kmeans.labels_.tolist())),
            }, save_path / f"{client}.pt")


def train_vq():
    clients = np.arange(args.num_clients)
    client_loaders = get_fl_train_dataloader()

    client_books = []
    for client in clients:
        book = torch.load(
            args.check_path / "codebooks" / f"{client}.pt",
            map_location=args.device,
            weights_only=True,
        )
        client_books.append(book)

    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    num_epochs_per_round = math.ceil(args.client_epochs / args.rounds)

    global_model = get_diff_model(in_channels=5)

    train_scheduler = get_train_scheduler()

    check_path = args.check_path / "vq"
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
            local_model = get_diff_model(in_channels=5)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = get_optimizer(local_model)
            client_loader = client_loaders[client]
            warmup_steps = get_warmup_steps(client_loader)
            client_book = client_books[client]
            avg_loss = 0
            for epoch in range(0, num_epochs_per_round):
                update_step = ((cur_round - 1) * num_epochs_per_round + epoch) * len(client_loader)
                total_loss = 0.0
                for x0, _, gidx, _ in client_loader:
                    x0 = x0.to(args.device)
                    B = x0.shape[0]

                    code_ids = torch.tensor(
                        [client_book["idxs"][idx] for idx in gidx.tolist()],
                        dtype=torch.long,
                        device=args.device,
                    )
                    condition = client_book["cen"][code_ids].reshape(B, 2, 32, 32)

                    t = torch.randint(0, args.T, (B,), device=args.device).long()
                    noise = torch.randn_like(x0)
                    xt = train_scheduler.add_noise(x0, noise, t)

                    model_input = torch.cat((xt, condition), dim=1)

                    noise_pred = local_model(model_input, t).sample
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
            logger.info(f"Round {cur_round} - Rank {i} - Client {client} Loss = {avg_loss:.4f}")
            client_state = local_model.state_dict()
            global_state = weight_aggregate(global_state, client_state, weights[i])

            del local_model
            del optimizer

        torch.save({
            "round": cur_round,
            "model": global_state,
        }, check_path / f"{cur_round}.pth")

        global_model.load_state_dict(global_state)

    torch.save(global_model.state_dict(), check_path / f"vq.pth")


if __name__ == "__main__":
    if args.method == "codebook":
        prepare_codebooks()
    else:
        train_vq()
