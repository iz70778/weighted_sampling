#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD"


num_clients=1
test_batchsize=32

for ((client_id=0; client_id<num_clients; client_id++)); do
    python -u -m synthesis.ws \
        --num_clients "$num_clients" \
        --test_batchsize "$test_batchsize" \
        --server mine
done

echo "All tasks completed."


