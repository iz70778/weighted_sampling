import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from config import args, global_model, linear_scheduler

if __name__ == "__main__":
    img_path = f"{args.results_path}/imgs/cl"
    os.makedirs(img_path, exist_ok=True)

    model = global_model
    model.eval()

    model.load_state_dict(
        torch.load(f"{args.results_path}/models/cl/model.pth", )
    )

    with torch.no_grad():
        for i in tqdm(range(0, args.num_samplings, args.test_batchsize)):

            n = min(args.test_batchsize, args.num_samplings - i)

            x = torch.randn(n, 3, args.image_size, args.image_size, device=args.device)

            for t in linear_scheduler.timesteps:
                noise_pred = model(x, t).sample
                x = linear_scheduler.step(noise_pred, t, x).prev_sample

            x = (x.clamp(-1, 1) + 1) / 2

            for j, img in enumerate(x):
                save_image(img, f"{img_path}/{i + j:05d}.png")