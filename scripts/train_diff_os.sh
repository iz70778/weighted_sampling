#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD"

num_clients=5
num_workers=2
server=mine

# Select scripts to run
scripts=(
    "os"
    # "cl"
)

# Shared task scheduler
run_task() {
    while (( $(jobs -rp | wc -l) >= num_workers )); do
        wait -n
    done

    "$@" &
}

for script in "${scripts[@]}"; do
    if [[ "$script" == "cl" ]]; then
        run_task python -u -m train.train_cl \
              --num_clients 1 \
              --suffix_path cl \
              --server "$server"
    elif [[ "$script" == "os" ]]; then
        for ((client_id=0; client_id<num_clients; client_id++)); do
            run_task python -u -m train.train_fed_os \
                --client_id "$client_id" \
                --suffix_path os \
                --server "$server"
        done
    fi
done

wait
echo "All tasks completed."