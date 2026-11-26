import logging
import os
import sys

import numpy as np
from diffusers import UNet2DModel, DDPMScheduler

from config import args

class Logger(logging.Logger):
    def __init__(self, name="ws"):
        super().__init__(name, level=logging.INFO)
        self.propagate = False

        self.formatter = logging.Formatter("%(asctime)s - %(message)s")

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self.formatter)
        self.addHandler(handler)

    def add_log_path(self, log_path, file_name="train.log"):
        os.makedirs(log_path, exist_ok=True)

        handler = logging.FileHandler(
            f"{log_path}/{file_name}", mode="a", encoding="utf-8"
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(self.formatter)

        self.addHandler(handler)

logger = Logger()


def get_diff_model():
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        dropout=0.1,
        norm_num_groups=32,
        attention_head_dim=256,
        add_attention=True,
        resnet_time_scale_shift="default",
    ).to(args.device)

    return model


def get_train_scheduler():
    if args.dataset == "cifar10":
        train_scheduler = DDPMScheduler(
            num_train_timesteps=args.T,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=False
        )
        return train_scheduler


def get_weights(selected_clients, client_loaders):
    temp = np.array([len(client_loaders[client]) for client in selected_clients])
    return temp / temp.sum()


