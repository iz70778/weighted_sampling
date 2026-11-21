#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD"


test_batchsize=256

python -u -m synthesis.cl \
        --test_batchsize "$test_batchsize" \
        --server mine

echo "All tasks completed."


