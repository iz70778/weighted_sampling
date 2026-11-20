#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD"


num_clients=5
test_batchsize=256

python -u -m synthesis.dfd \
        --num_clients "$num_clients" \
        --test_batchsize "$test_batchsize" \
        --server mine

echo "All tasks completed."


