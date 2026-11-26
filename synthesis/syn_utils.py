import os

import torch
from diffusers import DDIMScheduler, EMAModel
from torchvision.utils import save_image
from tqdm import tqdm

from config import args
from common_utils import get_train_scheduler, get_diff_model, logger

from cleanfid import fid
from cleanfid.inception_torchscript import InceptionV3W


def ddpm_sample(model, img_path, num_samplings=args.num_samplings, show_tqdm=False):
    train_scheduler = get_train_scheduler()

    model.eval()
    with torch.no_grad():
        for i in tqdm(
                range(0, num_samplings, args.test_batchsize),
                disable=not show_tqdm
        ):

            n = min(args.test_batchsize, num_samplings - i)

            x = torch.randn(n, 3, args.image_size, args.image_size, device=args.device)

            for t in train_scheduler.timesteps:
                noise_pred = model(x, t).sample
                x = train_scheduler.step(noise_pred, t, x).prev_sample

            x = (x.clamp(-1, 1) + 1) / 2

            for j, img in enumerate(x):
                save_image(img, f"{img_path}/{i + j:05d}.png")


def ddim_sample(
        model,
        img_path,
        num_samplings=args.num_samplings,
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

            x = torch.randn(n, 3, args.image_size, args.image_size, device=args.device)

            for t in sample_scheduler.timesteps:
                noise_pred = model(x, t).sample
                x = sample_scheduler.step(noise_pred, t, x, eta=eta).prev_sample

            x = (x.clamp(-1, 1) + 1) / 2

            for j, img in enumerate(x):
                save_image(img, f"{img_path}/{i + j:05d}.png")


def sample_data(model, img_path, num_samplings, method):
    logger.info(f"Generating {num_samplings} images for {method}...")
    os.makedirs(img_path, exist_ok=True)
    if args.ddim:
        ddim_sample(model, img_path, num_samplings)
    else:
        ddpm_sample(model, img_path, num_samplings)


def load_trained_model(model_name):
    model = get_diff_model()
    checkpoint = torch.load(
        f"{args.model_path}/{model_name}.pth",
        map_location=args.device,
        weights_only=True,
    )

    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        ema_model = EMAModel(model.parameters())
        ema_model.load_state_dict(checkpoint["ema"])
        ema_model.copy_to(model.parameters())
    else:
        model.load_state_dict(checkpoint)

    return model


def compute_clean_fid(img_path):
    score = fid.compute_fid(
        img_path,
        dataset_name=args.dataset,
        dataset_res=args.image_size,
        dataset_split="train",
        num_workers=0,
    )
    return score
