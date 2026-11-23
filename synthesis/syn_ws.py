import os
import torch
from diffusers import EMAModel

from config import args
from common_utils import logger, get_diff_model, get_weights
from data.data_loder import get_fl_train_dataloader
from synthesis.syn_utils import ddpm_sample, ddim_sample

if __name__ == "__main__":
    img_path = f"{args.img_path}/ws"
    os.makedirs(img_path, exist_ok=True)

    model = get_diff_model()

    clients = list(range(args.num_clients))
    client_loaders = get_fl_train_dataloader()

    weights = get_weights(clients, client_loaders)
    # weights = [1 / args.num_clients for _ in range(args.num_clients)]

    for client in clients:
        os.makedirs(f"{img_path}/{client}", exist_ok=True)

        # model.load_state_dict(
        #     torch.load(f"{args.model_path}/client{client}_os.pth", )
        # )

        checkpoint = torch.load(
            f"{args.model_path}/client{client}_os.pth",
            map_location=args.device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"])
        ema_model = EMAModel(model.parameters())
        ema_model.load_state_dict(checkpoint["ema"])
        ema_model.copy_to(model.parameters())

        num_samplings = int(weights[client] * args.num_samplings)
        logger.info(f"Generating {num_samplings} images for client {client}...")

        if args.ddim:
            ddim_sample(model, f"{img_path}/{client}", num_samplings)
        else:
            ddpm_sample(model, f"{img_path}/{client}", num_samplings)
