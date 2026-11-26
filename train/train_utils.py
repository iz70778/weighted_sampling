from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import EMAModel

import os
from config import args


def get_ema_model(model):
    return EMAModel(
        model.parameters(),
        decay=0.9999,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75,
    )


def get_warmup_steps(train_loader):
    total_updates = args.client_epochs * len(train_loader)
    warmup_ratio = 5000 / 800000
    warmup_steps = round(total_updates * warmup_ratio)
    return warmup_steps


def get_checkpoint(check_path):
    checkpoints = list(Path(check_path).glob("*.pth"))

    if len(checkpoints) == 0:
        return None

    latest_ckpt = max(
        checkpoints,
        key=lambda p: int(p.stem.rsplit("_", 1)[-1])
    )

    checkpoint = torch.load(
        latest_ckpt,
        map_location=args.device,
        weights_only=True,
    )

    return checkpoint


def save_model(model, ema_model, method):
    os.makedirs(args.model_path, exist_ok=True)

    ema_model.copy_to(model.parameters())

    torch.save(
        model.state_dict(),
        f"{args.model_path}/{method}.pth",
    )


def get_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return optimizer


def train_epoch(model, optimizer, train_loader,
                train_scheduler, update_step):
    model.train()
    total_loss = 0.0
    warmup_steps = get_warmup_steps(train_loader)

    for x0, _, _, _ in train_loader:
        x0 = x0.to(args.device)
        B = x0.shape[0]

        t = torch.randint(
            0, args.T, (B,), device=args.device
        ).long()

        noise = torch.randn_like(x0)
        xt = train_scheduler.add_noise(x0, noise, t)

        noise_pred = model(xt, t).sample
        loss = F.mse_loss(noise_pred, noise)

        # Learning rate warmup
        update_step += 1
        lr = args.lr * min(update_step / warmup_steps, 1.0)

        for group in optimizer.param_groups:
            group["lr"] = lr

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader), update_step


def weight_aggregate(global_state, local_state, weight):
    if global_state is None:
        global_state = {k: local_state[k] * weight for k in local_state}
    else:
        for k in global_state:
            global_state[k] += local_state[k] * weight
    return global_state