import math

import torch
from torch.utils.data import Subset, DataLoader, ConcatDataset

from common_utils import get_train_scheduler
from common_utils import logger, get_diff_model
from config import args
from data.data_loder import get_fl_train_dataloader
from train.train_utils import get_optimizer, get_ema_model, save_model, get_warmup_steps

if __name__ == '__main__':
    method = "pfdm"

    args.client_epochs = int(args.client_epochs * 0.3)

    model = get_diff_model()

    clients = list(range(args.num_clients))
    client_loaders = get_fl_train_dataloader()

    subsets = []
    for client_loader in client_loaders:
        client_dataset = client_loader.dataset
        train_dataset_size = int(len(client_dataset) * 0.3)
        indices = torch.randperm(len(client_dataset))[:train_dataset_size].tolist()
        subsets.append(Subset(client_dataset, indices))

    scheduler = get_train_scheduler()

    radius, depth = (15, 740) if args.dataset == "cifar10" else (35, 850)
    alpha = float(scheduler.alphas_cumprod[depth - 1])
    signal_scale = math.sqrt(alpha)
    noise_scale = math.sqrt(1 - alpha)
    ratio = 2 * radius * signal_scale / noise_scale
    epsilon = ratio ** 2 / 2 + ratio * math.sqrt(2 * math.log(1 / 1e-5))

    release_loader = DataLoader(
        ConcatDataset(subsets), batch_size=args.cl_batchsize, shuffle=False, num_workers=0
    )
    release_generator = torch.Generator(device="cpu")
    release_generator.seed()
    noisy_images = None
    offset = 0
    for images, *_ in release_loader:
        norms = images.flatten(1).norm(dim=1).clamp_min(1e-12)
        scale = (radius / norms).clamp_max(1).view(-1, 1, 1, 1)
        noise = torch.randn(images.shape, dtype=images.dtype, generator=release_generator)
        noisy_batch = signal_scale * images * scale + noise_scale * noise
        if noisy_images is None:
            noisy_images = images.new_empty((len(release_loader.dataset), *images.shape[1:]))
        noisy_images[offset: offset + len(images)] = noisy_batch
        offset += len(images)

    train_loader = DataLoader(
        noisy_images, batch_size=args.cl_batchsize, shuffle=True, num_workers=0
    )

    model = get_diff_model()
    optimizer = get_optimizer(model)
    ema_model = get_ema_model(model)
    warmup_steps = get_warmup_steps(train_loader)
    train_scheduler = get_train_scheduler()

    update_step = 0

    num_epochs = int(args.client_epochs * 0.3)

    model.train()
    for epoch in range(1, args.client_epochs + 1):
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(args.device)
            B = batch.shape[0]
            t = torch.randint(0, args.T, (B,), device=args.device).long()
            noise = torch.randn_like(batch)
            xt = train_scheduler.add_noise(batch, noise, t)
            noise_pred = model(xt, t).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            update_step += 1
            lr = args.lr * min(update_step / warmup_steps, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            ema_model.step(model.parameters())
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Method {method} - Epoch {epoch}: Loss = {avg_loss:.4f}")

    save_model(model, ema_model, method)
