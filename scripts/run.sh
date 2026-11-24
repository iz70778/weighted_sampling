#!/usr/bin/env bash
set -eo pipefail

# Load common configuration
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

parse_args "$@"
setup_environment

# Run one Python task
exec python -u -m "$script" \
    "${server_args[@]}" \
    "${python_args[@]}"
