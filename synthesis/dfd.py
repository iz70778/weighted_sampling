import io
import os
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from config import args, linear_scheduler, global_model


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

    client_models = []
    for client in range(args.num_clients):
        model = deepcopy(global_model)
        model.load_state_dict(
            torch.load(
                f"{args.results_path}/models/os/client{client}.pth",
                map_location="cpu"
            )
        )
        model.to(args.device)
        model.eval()
        client_models.append(model)

    linear_scheduler.set_timesteps(args.T)

    with torch.no_grad():
        for start in tqdm(range(0, args.num_samplings, args.test_batchsize), desc="Generating"):
            current_batch_size = min(args.test_batchsize, args.num_samplings - start)

            x = torch.randn(
                current_batch_size, 3, args.image_size, args.image_size, device=args.device
            )

            for t in linear_scheduler.timesteps:
                noises = []
                for model in client_models:
                    noise_pred = model(x, t).sample
                    noises.append(noise_pred)

                x = aggregate_fit(x, t, noises)

            x = (x.clamp(-1, 1) + 1) / 2

            for j, img in enumerate(x):
                save_image(img, f"{img_path}/{start + j:05d}.png")
