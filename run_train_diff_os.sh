#!/bin/bash
set -e

export PYTHONHASHSEED=0
export PYTHONPATH="$PWD"

seed=0
dataset="cifar10"
dirichlet_alpha=0.1
num_clients=1

for ((client_id=0; client_id<num_clients; client_id++)); do
    python -u scripts/train_diff_os.py \
        --seed "$seed" \
        --dataset "$dataset" \
        --dirichlet_alpha "$dirichlet_alpha" \
        --num_clients "$num_clients" \
        --client_id "$client_id" \
        --server mine
done

echo "All tasks completed."