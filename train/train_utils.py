import os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import EMAModel

from common_utils import logger, get_diff_model, get_train_scheduler
from config import args


def train_diff_model(train_dataloader, model_name):
    model = get_diff_model()

    train_scheduler = get_train_scheduler()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    # ema
    ema_model = EMAModel(
        model.parameters(),
        decay=0.9999,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75,
    )

    # warmup lr
    total_updates = args.client_epochs * len(train_dataloader)
    warmup_ratio = 5000 / 800000
    warmup_steps = max(1, round(total_updates * warmup_ratio))
    update_step = 0

    # load checkpoints
    check_path = f"{args.check_path}/{model_name}"
    logger.add_log_path(check_path)
    os.makedirs(check_path, exist_ok=True)
    checkpoints = list(Path(check_path).glob("*.pth"))
    if len(checkpoints) > 0:
        latest_ckpt = max(
            checkpoints,
            key=lambda p: int(p.stem.rsplit("_", 1)[-1])
        )

        checkpoint = torch.load(latest_ckpt, map_location=args.device, weights_only=True)

        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        update_step = checkpoint["update_step"]

        logger.info(f"Model {model_name} start training from epoch {start_epoch}")
    else:
        logger.info(f"Model {model_name} start training from scratch")
        start_epoch = 0

    for epoch in range(start_epoch, args.client_epochs):
        model.train()
        total_loss = 0.0

        for x0, _, _, _ in train_dataloader:
            x0 = x0.to(args.device)
            B = x0.shape[0]

            t = torch.randint(
                0, args.T, (B,), device=args.device
            ).long()

            noise = torch.randn_like(x0)
            xt = train_scheduler.add_noise(x0, noise, t)

            noise_pred = model(xt, t).sample
            loss = F.mse_loss(noise_pred, noise)

            update_step += 1
            lr = args.lr * min(update_step / warmup_steps, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            ema_model.step(model.parameters())

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Model {model_name} - Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        if (epoch + 1) % 100 == 0 or (epoch + 1) == args.client_epochs:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update_step": update_step,
            }, f"{check_path}/{model_name}_{epoch + 1}.pth")

        if (epoch + 1) == args.client_epochs:
            os.makedirs(args.model_path, exist_ok=True)
            ema_model.copy_to(model.parameters())
            torch.save(
                model.state_dict(),
                f"{args.model_path}/{model_name}.pth",
            )
    del model
    del optimizer
