from copy import deepcopy
import os
import torch
from tqdm import tqdm
from config import args
from train import client
from torchvision.utils import save_image
from train.diff_utils import linear_scheduler
from scripts.init import client_loaders, global_model
from cleanfid import fid
import numpy as np
import torch


import io
import numpy as np
from dataclasses import dataclass
from train.diff_utils import linear_scheduler

@dataclass
class Parameters:
    tensors: list[bytes]
    tensor_type: str = "numpy.ndarray"


def ndarrays_to_parameters(ndarrays):
    tensors = []
    for array in ndarrays:
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        tensors.append(buffer.getvalue())
    return Parameters(tensors=tensors)


def parameters_to_ndarrays(parameters):
    return [
        np.load(io.BytesIO(tensor), allow_pickle=False)
        for tensor in parameters.tensors
    ]

def get_step_size(alpha_bar_t, rho, p):
    assert (p >= 1.0) and (rho <= 0.5)
    return rho * (1 - alpha_bar_t) ** p  # / (3 * 2**p)


def get_mixture_scores(coef_comp, grad_comp):
    k, n, c, h, w = grad_comp.shape
    return coef_comp.view(k, n, 1).mul(grad_comp.view(k, n, -1)).sum(0).view(n, c, h, w)


def ula_step(x, neg_gradlogp, eta):
    std = (2 * eta) ** 0.5
    x = x - neg_gradlogp.mul(eta) + torch.randn_like(neg_gradlogp).mul(std)
    return x


def aggregate_fit(x, t, noises):
    # Retrieve hyperparmeters
    alpha_bar_t = linear_scheduler.alphas_cumprod[t].item()
    rho = 0.1
    ord = 1.0
    # rho = 1
    # ord = 2
    eta_t = get_step_size(alpha_bar_t, rho, ord)

    # Aggregate
    epsilons = torch.stack(noises, dim=0)

    mix_grads = epsilons * 2 / (1 - alpha_bar_t) ** 0.5
    mix_energies = epsilons.norm(p=2, dim=(2, 3, 4)).pow(2)

    logits = -mix_energies
    mix_coefs = logits.softmax(dim=0)
    mixture_scores = get_mixture_scores(mix_coefs, mix_grads)

    x = ula_step(x, mixture_scores, eta_t)

    return x


if __name__ == "__main__":
    img_path = f"{args.results_path}/imgs/dfd"
    os.makedirs(img_path, exist_ok=True)

    batch_size = 100
    total_samples = 5000
    device = args.device

    # Load all client models
    client_models = []
    
    for client_id in range(args.num_clients):
        model = deepcopy(global_model)
        model.load_state_dict(
            torch.load(
                f"{args.results_path}/0.1/client{client_id}_ddpm_model.pth",
                map_location="cpu"
            )
        )
        model.to(device)
        model.eval()
        client_models.append(model)

    # Set diffusion timesteps
    linear_scheduler.set_timesteps(args.T)

    # Generate synthetic images
    with torch.no_grad():
        for start in tqdm(range(0, total_samples, batch_size), desc="Generating"):
            current_batch_size = min(batch_size, total_samples - start)

            # Initialize Gaussian noise at server
            x = torch.randn(
                current_batch_size, 3, 32, 32, device=device
            )

            # Cooperative sampling
            for t in linear_scheduler.timesteps:

                # Collect noise predictions from all clients
                noises = []
                for model in client_models:
                    noise_pred = model(x, t).sample
                    noises.append(noise_pred)

                # Server aggregates scores and performs ULA
                x = aggregate_fit(x, t, noises)

            # Save generated images
            x = (x.clamp(-1, 1) + 1) / 2

            for j, img in enumerate(x):
                save_image(img, f"{img_path}/{start + j:05d}.png")

    