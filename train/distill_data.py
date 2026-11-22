import os

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

from config import args
from common_utils import get_diff_model
from utils.common_utils import logger
from data.data_loder import get_fl_train_dataloader

import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)

if __name__ == '__main__':
    model = get_diff_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    ema_model = AveragedModel(
        model,
        avg_fn=lambda ema, current, _: (
                0.9999 * ema + 0.0001 * current
        )
    )
    ema_model.update_parameters(model)

    client_loaders  = get_fl_train_dataloader()
    client_loader = client_loaders[client]

    model_path = f"{args.results_path}/models/data_distill"
    os.makedirs(model_path, exist_ok=True)

    total_updates = args.client_epochs * len(client_loader)
    warmup_ratio = 5000 / 800000
    warmup_steps = max(1, round(total_updates * warmup_ratio))
    update_step = 0

    for epoch in range(args.client_epochs):
        model.train()
        total_loss = 0.0

        for x0, _, _, _ in client_loader:
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
            ema_model.update_parameters(model)

            total_loss += loss.item()
            # progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(client_loader)
        logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        if (epoch + 1) % 100 == 0:
            torch.save(
                ema_model.module.state_dict(),
                f"{model_path}/{epoch}.pth",
            )

    del model
    del optimizer
