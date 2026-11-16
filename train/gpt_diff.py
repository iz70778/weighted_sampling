
import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm

# ==========================
# 1. Configuration
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 32
batch_size = 128
epochs = 50
lr = 1e-4
T = 1000

os.makedirs("outputs", exist_ok=True)

# ==========================
# 2. Dataset
# ==========================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3)
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device == "cuda")
)

# ==========================
# 3. Initialize DDPM
# ==========================

model = UNet2DModel(
    sample_size=image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    ),
    norm_num_groups=32,
).to(device)

scheduler = DDPMScheduler(
    num_train_timesteps=T,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr
)

# ==========================
# 4. Training
# ==========================

for epoch in range(epochs):

    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

    for x0, _ in progress:

        x0 = x0.to(device)
        B = x0.shape[0]

        # Step 1: Random timesteps
        t = torch.randint(
            0, T, (B,), device=device
        ).long()

        # Step 2: Sample Gaussian noise
        noise = torch.randn_like(x0)

        # Step 3: Forward diffusion
        xt = scheduler.add_noise(x0, noise, t)

        # Step 4: Predict noise
        noise_pred = model(xt, t).sample

        # Step 5: Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # Step 6: Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Save model after every epoch
    torch.save(
        model.state_dict(),
        "outputs/ddpm_model.pth"
    )

# ==========================
# 5. Generate Images
# ==========================

model.eval()

scheduler.set_timesteps(T)

with torch.no_grad():

    # Start from pure Gaussian noise
    x = torch.randn(
        16, 3, image_size, image_size,
        device=device
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
