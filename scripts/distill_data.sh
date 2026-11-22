#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD"

test_batchsize=256

server=mine

# generate data
python -u -m synthesis.syn_ws \
        --test_batchsize "$test_batchsize" \
        --num_samplings 50000 \
        --server "$server"

wait


# train on generated data

