import os

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

from config import args, global_model, client_loaders, linear_scheduler, logger

if __name__ == '__main__':
    client = args.client_id

    client_model = global_model

    optimizer = torch.optim.Adam(
        client_model.parameters(),
        lr=args.lr
    )

    client_loader = client_loaders[client]

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

            optimizer.step()

            total_loss += loss.item()
            # progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(client_loader)
        logger.info(f"Client {client} - Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        model_path = f"{args.results_path}/models/os"
        os.makedirs(model_path, exist_ok=True)

        if (epoch + 1) % 100 == 0:
            torch.save(
                client_model.state_dict(),
                f"{model_path}/client{client}_{epoch}.pth",
            )

    del client_model
    del optimizer
