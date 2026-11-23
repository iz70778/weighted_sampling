#!/bin/bash
set -e

# Arguments and defaults
num_clients=${1:-5}
num_workers=${2:-2}
server=${3:-mine}
scripts=("${@:4}")

if (( ${#scripts[@]} == 0 )); then
    scripts=("cl")
fi

# Configure the environment based on server
case "$server" in
    mine)
        conda_env="ttt"
        ;;
    calder)
        module load calder/hopper miniforge3
        conda_env="ttt"
        ;;
    aire)
        module load miniforge
        conda_env="ttt"
        ;;
    *)
        echo "Unknown server: $server. Use mine, calder, or aire." >&2
        exit 1
        ;;
esac

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$conda_env"

export PYTHONHASHSEED=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

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
    else
        echo "Unknown script: $script" >&2
        exit 1
    fi
done

wait
echo "All tasks completed."