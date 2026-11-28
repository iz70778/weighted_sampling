import os

import torch
from diffusers import DDIMScheduler
from torchvision.utils import save_image
from tqdm import tqdm

from common_utils import get_train_scheduler, logger
from config import args
from data.data_loder import get_fl_train_dataloader
from synthesis.syn_utils import load_trained_model


def ddpm_sample(model, img_path, num_samplings=args.num_samplings, code_book=None, show_tqdm=False):
    train_scheduler = get_train_scheduler()

    model.eval()
    with torch.no_grad():
        for i in tqdm(
                range(0, num_samplings, args.test_batchsize),
                disable=not show_tqdm
        ):

            n = min(args.test_batchsize, num_samplings - i)

            centers, counts = codebook
            code_ids = torch.multinomial(counts, n, replacement=True)
            condition = centers[code_ids].reshape(n, 2, 32, 32)

            x = torch.randn(n, 3, args.image_size, args.image_size, device=args.device)

            for t in train_scheduler.timesteps:
                model_input = x if condition is None else torch.cat((x, condition), dim=1)
                noise_pred = model(model_input, t).sample
                x = train_scheduler.step(noise_pred, t, x).prev_sample

            x = (x.clamp(-1, 1) + 1) / 2

            for j, img in enumerate(x):
                save_image(img, img_path / f"{i + j:05d}.png")


def ddim_sample(
        model,
        img_path,
        num_samplings=args.num_samplings,
        codebook=None,
        num_inference_steps=50,
        eta=0.0,
        show_tqdm=False,
):
    train_scheduler = get_train_scheduler()

    sample_scheduler = DDIMScheduler.from_config(train_scheduler.config)
    sample_scheduler.set_timesteps(num_inference_steps)

    model.eval()

    with torch.no_grad():
        for i in tqdm(
                range(0, num_samplings, args.test_batchsize),
                disable=not show_tqdm
        ):
            n = min(args.test_batchsize, num_samplings - i)

            centers, counts = codebook
            code_ids = torch.multinomial(counts, n, replacement=True)
            condition = centers[code_ids].reshape(n, 2, 32, 32)

            x = torch.randn(n, 3, args.image_size, args.image_size, device=args.device)

            for t in sample_scheduler.timesteps:
                model_input = x if condition is None else torch.cat((x, condition), dim=1)
                noise_pred = model(model_input, t).sample
                x = sample_scheduler.step(noise_pred, t, x, eta=eta).prev_sample

            x = (x.clamp(-1, 1) + 1) / 2

            for j, img in enumerate(x):
                save_image(img, img_path / f"{i + j:05d}.png")


def sample_data(model, img_path, num_samplings, codebook):
    os.makedirs(img_path, exist_ok=True)
    if args.ddim:
        ddim_sample(model, img_path, num_samplings, codebook)
    else:
        ddpm_sample(model, img_path, num_samplings, codebook)


if __name__ == "__main__":
    method = "vq"

    img_path = args.img_path / method
    os.makedirs(img_path, exist_ok=True)

    clients = list(range(args.num_clients))
    client_loaders = get_fl_train_dataloader()

    weights = [1 / args.num_clients for _ in range(args.num_clients)]

    for client in clients:
        book = torch.load(
            args.check_path / "codebooks" / f"{client}.pt",
            map_location=args.device,
            weights_only=True,
        )
        centers = book["cen"]
        idxs = torch.tensor(list(book["idxs"].values()), dtype=torch.long)
        counts = torch.bincount(idxs, minlength=len(centers))
        codebook = centers, counts.to(device=centers.device, dtype=torch.float32)

        model = load_trained_model(f"client{client}_os", in_channels=5)
        num_samplings = int(weights[client] * args.num_samplings)
        logger.info(f"Generating {num_samplings} images for Client {client}...")
        sample_data(model, img_path / str(client), num_samplings, codebook)
