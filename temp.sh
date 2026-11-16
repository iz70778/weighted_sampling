
#!/bin/bash
#SBATCH --job-name=weighted_sampling
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/users/sc21s2j/weighted_sampling/outputs/slurm-%j.out

module load calder/hopper miniforge3
conda activate ws

cd /users/sc21s2j/weighted_sampling

python -u run_train_diff_os.py


