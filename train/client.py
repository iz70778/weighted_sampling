import math
import copy
import torch
from torch import device, nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from args import args
from train import model
from diffusers import DDPMScheduler
import torch.nn.functional as F
from torchvision.utils import save_image

criterion = nn.CrossEntropyLoss()


def copy_global_model(global_model):
    global_model_copy = copy.deepcopy(global_model)
    global_model_copy.to(args.device)
    return global_model_copy

def get_lr(current_round):
    if args.dataset in ['mnist', 'cifar10', 'fmnist', 'cinic10']:
        return pow(0.99, current_round - 1) * args.lr
    elif args.dataset in ['cifar100', 'tiny_imagenet']:
        cos_inner = math.pi * current_round / args.rounds
        return 0.5 * args.lr * (1 + math.cos(cos_inner))
    else:
        raise 'lr decay strategy is not set'


def train_client(client_loader, global_model, cur_round):
    # global model is on gpu
    client_model = copy_global_model(global_model)
    client_model.train()
    lr = get_lr(cur_round)
    optimizer = torch.optim.SGD(client_model.parameters(), lr)
    total = 0
    for epoch in range(args.client_epochs):
        for x, y, _, _ in client_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            out = client_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += x.shape[0]
    local_state = client_model.state_dict()

    # free cache
    del client_model
    del optimizer

    return local_state

scheduler = DDPMScheduler(
    num_train_timesteps=args.T,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon"
)

def train_diff_os(client, client_loader, global_model,):
    client_model = copy_global_model(global_model)
    optimizer = torch.optim.AdamW(
        client_model.parameters(),
        lr=args.lr
    )
    for epoch in range(args.client_epochs):
        client_model.train()
        total_loss = 0.0
        progress = tqdm(client_loader, desc=f"Epoch {epoch+1}/{args.client_epochs}")
        for x0, _, _, _ in  progress:
            x0 = x0.to(args.device)
            B = x0.shape[0]
            t = torch.randint(
                0, args.T, (B,), device=args.device
            ).long()
            noise = torch.randn_like(x0)
            xt = scheduler.add_noise(x0, noise, t)
            noise_pred = client_model(xt, t).sample
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(client_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Save model after every epoch
        torch.save(
            client_model.state_dict(),
             f"/scratch-calder/users/sc21s2j/ws/results/client{client}_ddpm_model.pth"
        )

    del client_model
    del optimizer

def test_diff(model):
    model.eval()

    scheduler.set_timesteps(args.T)

    with torch.no_grad():

        # Start from pure Gaussian noise
        x = torch.randn(
            16, 3, args.image_size, args.image_size,
            device=args.device
        )

        for t in tqdm(scheduler.timesteps, desc="Sampling"):

            noise_pred = model(x, t).sample

            x = scheduler.step(
                noise_pred, t, x
            ).prev_sample

        # Convert [-1, 1] to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2

        save_image(
            x,
            "outputs/generated.png",
            nrow=4
        )

    print("Training and sampling completed!")
