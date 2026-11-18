
from pathlib import Path
from diffusers import DDPMScheduler
from cleanfid import fid
from config import args

linear_scheduler = DDPMScheduler(
    num_train_timesteps=args.T,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon"
)

