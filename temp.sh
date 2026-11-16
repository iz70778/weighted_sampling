
#!/bin/bash
#SBATCH --job-name=fed_ddpm
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/users/sc21s2j/fed_diff/slurm-%j.out

set -e

# Set Calder software environment
export MODULEPATH=/uol/apps/calder/modules

module load calder/hopper miniforge3
conda activate diffusion

# Go to project directory
cd /users/sc21s2j/fed_diff

# Start 10 clients with 3 concurrent processes
python -u launcher.py \
    --num-clients 10 \
    --num-workers 3
