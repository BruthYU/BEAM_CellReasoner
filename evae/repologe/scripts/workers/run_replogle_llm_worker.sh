#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_dir="$project_dir/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
cd "$project_dir"
export PATH="/opt/sc-evae/bin:$PATH"
export PYTHONPATH="$artifact_dir/environment/python_packages:$artifact_dir/source_overlay"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$artifact_dir/hf_cache" TRITON_CACHE_DIR="$artifact_dir/triton_cache"
export WANDB_MODE=offline
cleanup() {
    code=$?
    printf '%s\t%s\n' "$(date -Iseconds)" "$code" >> "$artifact_dir/worker_exit.tsv"
    if [[ "$(id -u)" == 0 ]]; then
        chown -Rh "$(stat -c '%u:%g' "$project_dir")" "$artifact_dir" "$project_dir/outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
    fi
}
trap cleanup EXIT
/opt/sc-evae/bin/python -u scripts/training/run_replogle_llm_pipeline.py 2>&1 | tee "$artifact_dir/worker.log"
