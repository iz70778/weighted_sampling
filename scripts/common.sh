#!/usr/bin/env bash

COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$COMMON_DIR/.." && pwd)"

# Parse named arguments
parse_args() {
    script=""
    server="mine"
    max_jobs=2
    python_args=()

    while (( $# > 0 )); do
        case "$1" in
            --script|--server|--max_jobs)
                if (( $# < 2 )) || [[ "$2" == --* ]]; then
                    echo "Missing value for $1" >&2
                    return 2
                fi

                case "$1" in
                    --script) script="$2" ;;
                    --server) server="$2" ;;
                    --max_jobs) max_jobs="$2" ;;
                esac
                shift 2
                ;;
            --)
                shift
                python_args+=("$@")
                break
                ;;
            *)
                python_args+=("$1")
                shift
                ;;
        esac
    done
}


# Set up server environment
setup_environment() {
    case "$server" in
        mine)
            results_path="F:/pycodes/weighted_sampling/results"
            dataset_path="F:/data"
            local conda_env="ttt"
            ;;
        calder)
            module load calder/hopper miniforge3
            results_path="/scratch-calder/users/sc21s2j/weighted_sampling/results"
            dataset_path="/scratch-calder/users/sc21s2j/dataset"
            local conda_env="ttt"
            ;;
        aire)
            module load miniforge
            results_path="/mnt/scratch/sc21s2j/weighted_sampling/results"
            dataset_path="/mnt/scratch/sc21s2j/dataset"
            local conda_env="ttt"
            ;;
        *)
            echo "Unknown server: $server" >&2
            return 2
            ;;
    esac

    if [[ "$server" != "mine" ]]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$conda_env"
    fi

    cd "$PROJECT_ROOT"

    export PYTHONHASHSEED=0
    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

    server_args=(
        --results_path "$results_path"
        --dataset_path "$dataset_path"
    )
}

# Concurrent task scheduler
run_task() {
    while (( running >= max_jobs )); do
        wait_one
    done

    "$@" &
    running=$((running + 1))
}

wait_one() {
    if ! wait -n; then
        failed=1
    fi
    running=$((running - 1))
}

wait_all() {
    while (( running > 0 )); do
        wait_one
    done

    if (( failed )); then
        echo "At least one task failed." >&2
        return 1
    fi

    echo "All tasks completed."
}
