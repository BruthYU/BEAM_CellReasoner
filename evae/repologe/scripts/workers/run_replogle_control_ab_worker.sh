#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
variant="${1:?A, B or C required}"
gpus="${2:?GPU count required}"
stage="${3:?smoke, benchmark or train required}"
run_id="${4:?Unique run ID required}"
[[ "$variant" == A || "$variant" == B || "$variant" == C ]]
[[ "$gpus" == 2 || "$gpus" == 4 ]]
[[ "$stage" == smoke || "$stage" == benchmark || "$stage" == train ]]
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
artifact_dir="$project_dir/artifacts/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-v1"
baseline_art="$project_dir/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
run_dir="$project_dir/outputs/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-v1/$variant/$run_id"
train_args=(scripts/training/train_replogle_control_ab.py --variant "$variant")
check_args=(scripts/checks/check_replogle_control_ab.py --verify-only)
if [[ "$variant" == C ]]; then
    n="${5:?Control count required}"; batch="${6:-2}"
    artifact_dir="$project_dir/artifacts/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-c-v1"
    run_dir="$project_dir/outputs/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-c-v1/n${n}/$run_id"
    train_args=(scripts/training/train_replogle_control_c.py --control-cells-nums "$n" --per-device-batch-size "$batch")
    check_args=(scripts/checks/check_replogle_control_c.py --verify-only --control-cells-nums "$n" --gpus "$gpus" --per-device-batch-size "$batch")
fi
# Refuse a second job with the same ID, even if the first failed partway.
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
export PATH="/opt/sc-evae/bin:$PATH"
export PYTHONPATH="$baseline_art/environment/python_packages:$baseline_art/source_overlay"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$((32 / gpus))" OPENBLAS_NUM_THREADS="$((32 / gpus))" MKL_NUM_THREADS="$((32 / gpus))"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export HF_HOME="$run_dir/hf_cache" TRITON_CACHE_DIR="$run_dir/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$run_dir/inductor_cache" XDG_CACHE_HOME="$run_dir/xdg_cache"
cleanup() {
    code=$?
    printf '%s\t%s\n' "$(date -Iseconds)" "$code" >> "$run_dir/worker_exit.tsv"
    if [[ "$(id -u)" == 0 ]]; then
        chown -Rh "$(stat -c '%u:%g' "$project_dir")" "$run_dir"
    fi
}
trap cleanup EXIT
python "${check_args[@]}" 2>&1 | tee "$run_dir/preflight.log"
run_phase() {
    phase="$1"
    printf '%s\t%s\n' "$(date -Iseconds)" "$phase" > "$run_dir/worker_phase.tsv"
    torchrun --standalone --nnodes=1 "--nproc-per-node=$gpus" "${train_args[@]}" --gpus "$gpus" --phase "$phase" --output "$run_dir/$phase" \
        2>&1 | tee "$run_dir/$phase.log"
}
run_phase smoke
if [[ "$stage" == benchmark ]]; then run_phase benchmark; fi
if [[ "$stage" == train ]]; then run_phase formal; fi
printf '%s\tcomplete\n' "$(date -Iseconds)" > "$run_dir/worker_phase.tsv"
