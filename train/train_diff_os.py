import logging
import os

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

from config import args, global_model, client_loaders, linear_scheduler

if __name__ == '__main__':
    client = args.client_id

    client_model = global_model
    ema_model = AveragedModel(
        client_model,
        avg_fn=lambda ema, current, _: 0.9999 * ema + 0.0001 * current,
    )
    ema_model.update_parameters(client_model)

    optimizer = torch.optim.Adam(
        client_model.parameters(),
        lr=args.lr
    )

    client_loader = client_loaders[client]

    total_updates = args.client_epochs * len(client_loader)
    warmup_steps = max(1, round(args.warmup_ratio * total_updates))
    update_step = 0

    for epoch in range(args.client_epochs):
        client_model.train()
        total_loss = 0.0
        # progress = tqdm(client_loader, desc=f"Epoch {epoch+1}/{args.client_epochs}")
        for x0, _, _, _ in client_loader:
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

            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)

            update_step += 1
            for group in optimizer.param_groups:
                group["lr"] = args.lr * min(update_step / warmup_steps, 1.0)

            optimizer.step()
            ema_model.update_parameters(client_model)

            total_loss += loss.item()
            # progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(client_loader)
        logging.info(f"Client {client} - Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        model_path = f"{args.results_path}/models/os"
        os.makedirs(model_path, exist_ok=True)

        if (epoch + 1) % 100 == 0:
            torch.save(
                ema_model.module.state_dict(),
                f"{model_path}/client{client}_{epoch}.pth",
            )

    del client_model
    del optimizer
