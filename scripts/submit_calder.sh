#!/bin/bash
#SBATCH --job-name=ws_train
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

# Forward all arguments to your training script
bash scripts/train_diff_os.sh "$@"