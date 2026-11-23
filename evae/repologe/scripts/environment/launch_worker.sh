#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${SC_EVAE_IMAGE:-registry.h.pjlab.org.cn/ailab-beam-beam_gpu/aivc-celltok:sc-evae-py312-torch291-cu128-v1}"
options=(
    --gpu=1
    --cpu=8
    --memory=64000
    --charged-group=beam_gpu_pool
    "--image=$image"
    --mount=gpfs://gpfs2/beam-gpfs02/wangyue:/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue
    "--workdir=$project_dir"
    "--max-wait-duration=${SC_EVAE_MAX_WAIT:-10m}"
)
case "${1:-}" in
    "") exec rlaunch "${options[@]}" -- bash ;;
    --check)
        exec rlaunch "${options[@]}" -- bash -c \
            'exec /opt/sc-evae/bin/python "$1/scripts/environment/check_environment.py" --require-cuda' \
            bash "$project_dir"
        ;;
    --dry-run) printf '%q ' rlaunch "${options[@]}" -- bash; printf '\n' ;;
    *) printf 'Usage: %s [--check|--dry-run]\n' "$0" >&2; exit 2 ;;
esac
