import torch
from diffusers import DDIMScheduler
from torchvision.utils import save_image
from tqdm import tqdm

from config import args
from common_utils import get_train_scheduler


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

            x = torch.randn(
                n, 3, args.image_size, args.image_size, device=args.device
            )

            for t in sample_scheduler.timesteps:
                noise_pred = model(x, t).sample
                x = sample_scheduler.step(noise_pred, t, x, eta=eta).prev_sample

            x = (x.clamp(-1, 1) + 1) / 2

            for j, img in enumerate(x):
                save_image(img, f"{img_path}/{i + j:05d}.png")


# mean = torch.tensor(
#     [0.4914, 0.4822, 0.4465],
#     device=x.device
# ).view(1, 3, 1, 1)
#
# std = torch.tensor(
#     [0.2023, 0.1994, 0.2010],
#     device=x.device
# ).view(1, 3, 1, 1)
#
# x = (x * std + mean).clamp(0, 1)