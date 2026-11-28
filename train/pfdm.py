import math

import torch
from torch.utils.data import Subset, DataLoader, ConcatDataset

from common_utils import get_train_scheduler
from common_utils import logger, get_diff_model
from config import args
from data.data_loder import get_fl_train_dataloader
from train.train_utils import get_optimizer, get_ema_model, save_ema_model, get_warmup_steps, get_checkpoint

if __name__ == '__main__':
    method = "pfdm"

    epoch_ratio = 0.3
    data_ratio = 0.3

    args.client_epochs = int(args.client_epochs * epoch_ratio)

    model = get_diff_model()

    clients = list(range(args.num_clients))
    client_loaders = get_fl_train_dataloader()

    train_scheduler = get_train_scheduler()

    radius, depth = (15, 740) if args.dataset == "cifar10" else (35, 850)
    alpha = float(train_scheduler.alphas_cumprod[depth - 1])
    signal_scale = math.sqrt(alpha)
    noise_scale = math.sqrt(1 - alpha)
    ratio = 2 * radius * signal_scale / noise_scale
    epsilon = ratio ** 2 / 2 + ratio * math.sqrt(2 * math.log(1 / 1e-5))

    # construct noisy dataset for server model
    subsets = []
    for client_loader in client_loaders:
        client_dataset = client_loader.dataset
        train_dataset_size = int(len(client_dataset) * data_ratio)
        indices = torch.randperm(len(client_dataset))[:train_dataset_size].tolist()
        subsets.append(Subset(client_dataset, indices))
    release_loader = DataLoader(
        ConcatDataset(subsets), batch_size=args.cl_batchsize, shuffle=False, num_workers=0
    )
    noisy_batches = []
    for images, *_ in release_loader:
        norms = images.flatten(1).norm(dim=1).clamp_min(1e-12)
        scale = (radius / norms).clamp_max(1).view(-1, 1, 1, 1)
        noisy_batches.append(
            signal_scale * images * scale + noise_scale * torch.randn_like(images)
        )
    train_loader = DataLoader(
        torch.cat(noisy_batches),
        batch_size=args.cl_batchsize,
        shuffle=True,
    )
    del noisy_batches

    model = get_diff_model()
    optimizer = get_optimizer(model)
    ema_model = get_ema_model(model)
    warmup_steps = get_warmup_steps(train_loader)

    check_path = args.check_path / method
    check_path.mkdir(parents=True, exist_ok=True)
    logger.add_log_path(check_path)
    checkpoint = get_checkpoint(check_path)
    if checkpoint is None:
        logger.info(f"Method {method} starts training from scratch")
        start_epoch = 0
        update_step = 0
    else:
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"]
        update_step = checkpoint["update_step"]

        if start_epoch >= args.client_epochs:
            logger.info(f"Method {method} has completed training")
        else:
            logger.info(f"Method {method} start training "
                        f"from epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch + 1, args.client_epochs + 1):
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
        if epoch % 100 == 0 or epoch == args.client_epochs:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update_step": update_step,
            }, f"{check_path}/{method}_{epoch}.pth")

    ema_model.copy_to(model.parameters())
    torch.save(
        model.state_dict(),
        args.model_path / f"{method}.pth",
    )

