#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD"

test_batchsize=256
num_samplings=5000
server=mine

# Select scripts to run
scripts=(
    "ws"
#    "cl"
#    "dfd"
#    "avg_merge"
)

for script in "${scripts[@]}"; do
    echo "Running synthesis.${script}..."

    python -u -m "synthesis.syn_${script}" \
        --test_batchsize "$test_batchsize" \
        --num_samplings "$num_samplings" \
        --server "$server"
done

echo "All tasks completed."

