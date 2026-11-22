import logging
import sys

import numpy as np
from diffusers import UNet2DModel, DDPMScheduler

from config import args


def init_logger():
    logger = logging.getLogger("ws")
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return logger


logger = init_logger()


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


def weight_aggregate(global_state, local_state, weight):
    if global_state is None:
        global_state = {k: local_state[k] * weight for k in local_state}
    else:
        for k in global_state:
            global_state[k] += local_state[k] * weight
    return global_state


def weights_aggregate(client_states, clients, weights):
    global_state = None
    for i, client in enumerate(clients):
        global_state = weight_aggregate(global_state, client_states[client], weights[i])
    return global_state
