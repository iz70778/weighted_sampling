import os
import torch
from config import args
from train import client
from torchvision.utils import save_image
from train.diff_utils import linear_scheduler
from scripts.init import client_loaders, global_model
from cleanfid import fid

img_path = f"{args.results_path}/imgs/all"


def syn_client_imgs(client, model, num_samples, batch_size=100):
    model.eval()

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):

            n = min(batch_size, num_samples - i)

            x = torch.randn(n, 3, args.image_size, args.image_size, device=args.device)

            for t in linear_scheduler.timesteps:
                noise_pred = model(x, t).sample
                x = linear_scheduler.step(noise_pred, t, x).prev_sample

            x = (x.clamp(-1, 1) + 1) / 2

            for j, img in enumerate(x):
                save_image(img, f"{img_path}/{client}_{i+j:05d}.png")


if __name__ == "__main__":
    os.makedirs(img_path, exist_ok=True)
    batch_size = 100
    total_samples = 5000
    model = global_model
    client_num_samples = [
        len(client_loaders[client]) for client in range(args.num_clients)
    ]
    weights = [count / sum(client_num_samples) for count in client_num_samples]

    for client in range(args.num_clients):
        model.load_state_dict(
            torch.load(f"{args.results_path}/all_0.1/client{client}_ddpm_model.pth")
        )
        num_samples = int(weights[client] * total_samples)
        print(f"Generating {num_samples} images for client {client}...")
        syn_client_imgs(client, model, num_samples, batch_size)
