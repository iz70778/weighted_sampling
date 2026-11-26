import math

import torch
from torch.utils.data import Subset, default_collate, DataLoader, ConcatDataset

from common_utils import get_train_scheduler
from common_utils import logger, get_diff_model
from config import args
from data.data_loder import get_fl_train_dataloader
from train.train_utils import get_optimizer, get_ema_model, train_epoch, save_model

if __name__ == '__main__':
    method = "pfdm"

    args.client_epochs = int(args.client_epochs * 0.3)
    args.batch_size = args.cl_batch_size

    model = get_diff_model()

    clients = list(range(args.num_clients))
    client_loaders = get_fl_train_dataloader()

    subsets = []
    for client_loader in client_loaders:
        client_dataset = client_loader.dataset
        train_dataset_size = int(len(client_dataset) * 0.3)
        indices = torch.randperm(len(client_dataset))[:train_dataset_size].tolist()
        subsets.append(Subset(client_dataset, indices))

    radius, depth = (15, 740) if args.dataset == "cifar10" else (35, 850)
    scheduler = get_train_scheduler()
    alpha = float(scheduler.alphas_cumprod[depth - 1])
    sensitivity_over_sigma = 2 * radius * math.sqrt(alpha / (1 - alpha))
    epsilon = sensitivity_over_sigma ** 2 / 2 + sensitivity_over_sigma * math.sqrt(2 * math.log(1e5))


    def collate_noisy(samples):
        x, _, _, _ = default_collate(samples)

        norms = x.flatten(1).norm(dim=1).clamp_min(1e-12)
        x = x * (radius / norms).clamp_max(1).view(-1, 1, 1, 1)

        noise = torch.randn(x.shape, dtype=x.dtype)
        x = math.sqrt(alpha) * x + math.sqrt(1 - alpha) * noise
        return x


    train_loader = DataLoader(
        ConcatDataset(subsets),
        batch_size=client_loaders[0].batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_noisy,
    )

    optimizer = get_optimizer(model)
    ema_model = get_ema_model(model)
    train_scheduler = get_train_scheduler()
    num_epochs = int(args.client_epochs * 0.3)

    for epoch in range(1, args.client_epochs + 1):
        loss, update_step = train_epoch(model, optimizer, train_loader, train_scheduler, update_step)
        ema_model.step(model.parameters())
        logger.info(f"Method {method} - Epoch {epoch}: Loss = {loss:.4f}")

    save_model(model, ema_model, method)
