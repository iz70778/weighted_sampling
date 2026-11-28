import os

import torch
from diffusers import DDIMScheduler
from torchvision.utils import save_image
from tqdm import tqdm

from common_utils import get_weights, logger, get_train_scheduler
from config import args
from data.data_loder import get_fl_train_dataloader
from synthesis.syn_utils import load_trained_model, sample_data

if __name__ == "__main__":
    img_path = args.img_path / args.method
    img_path.mkdir(parents=True, exist_ok=True)

    clients = list(range(args.num_clients))

    global_model = load_trained_model(args.method).eval()

    depth = 740 if args.dataset == "cifar10" else 850

    scheduler = get_train_scheduler()

    stride = 1
    if args.ddim:
        ddim_steps = 50
        scheduler = DDIMScheduler.from_config(scheduler.config, set_alpha_to_one=True)
        scheduler.set_timesteps(ddim_steps)
        stride = args.T // ddim_steps

    with torch.no_grad():
        for client in range(args.num_clients):
            num_samplings = args.num_samplings // len(clients)
            client_model = load_trained_model(f"client{client}_os").to(args.device).eval()
            batches = range(0, num_samplings, args.test_batchsize)
            for i in tqdm(batches, desc=f"Client {client}"):
                n = min(args.test_batchsize, num_samplings - i)
                x = torch.randn(n, 3, args.image_size, args.image_size, device=args.device)

                start_t = args.T - 1
                for t in range(start_t, -1, -stride):
                    noise_pred = global_model(x, t).sample
                    x = scheduler.step(noise_pred, t, x).prev_sample

                start_t = depth - 1
                for t in range(start_t, -1, -stride):
                    noise_pred = client_model(x, t).sample
                    x = scheduler.step(noise_pred, t, x).prev_sample

                x = (x.clamp(-1, 1) + 1) / 2

                for j, img in enumerate(x):
                    save_image(img, img_path / f"{client}/{i + j:05d}.png")

            del client_model
