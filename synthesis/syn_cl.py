import os

import torch

from config import args
from common_utils import get_diff_model
from synthesis.syn_utils import ddpm_sample, ddim_sample

if __name__ == "__main__":
    img_path = f"{args.img_path}/cl"
    os.makedirs(img_path, exist_ok=True)

    model = get_diff_model()

    model.load_state_dict(
        torch.load(f"{args.model_path}/cl.pth", )
    )

    if args.ddim_sample:
        ddim_sample(model, img_path, show_tqdm=True)
    else:
        ddpm_sample(model, img_path, show_tqdm=True)
