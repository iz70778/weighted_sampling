#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD"

num_clients=5
num_workers=1

for ((client_id=0; client_id<num_clients; client_id++)); do
    python -u -m train.diff_os \
        --client_id "$client_id" \
        --server mine &
    if (( $(jobs -rp | wc -l) >= num_workers )); then
        wait -n
    fi
done

wait

#python -u scripts/train_diff_os.py \
#        --num_clients 1 \
#        --server mine &

echo "All tasks completed."