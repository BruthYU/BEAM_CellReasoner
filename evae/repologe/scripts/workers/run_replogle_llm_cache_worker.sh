#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_dir="$project_dir/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
cd "$project_dir"
export PYTHONPATH="$artifact_dir/source_overlay" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cleanup() {
    code=$?
    printf '%s\t%s\n' "$(date -Iseconds)" "$code" >> "$artifact_dir/cache_worker_exit.tsv"
    if [[ "$(id -u)" == 0 ]]; then chown -Rh "$(stat -c '%u:%g' "$project_dir")" "$artifact_dir/cache"; fi
}
trap cleanup EXIT
/opt/sc-evae/bin/python -u scripts/data/cache_replogle_nb_tokens.py 2>&1 | tee "$artifact_dir/cache_worker.log"
