#!/usr/bin/env bash
# 任务内部调用，需已有 GPU；日常提交用 scripts/launch/submit_replogle_control_eval.sh。
# bash scripts/workers/run_replogle_control_eval_worker.sh B NEW_RUN_ID 64
# 位置参数：A|B|C、新运行名、生成 batch（默认64）。

set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
variant="${1:?A, B or C required}"
run_id="${2:?Run ID required}"
batch_size="${3:-64}"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]]
[[ "$variant" == A || "$variant" == B || "$variant" == C ]]
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
run_dir="$project_dir/outputs/replogle_llm_control_evaluation/ab-observed-depth-v1/$variant/$run_id"
eval_args=(scripts/evaluation/eval_replogle_qwen.py --protocol legacy-control --variant "$variant")
if [[ "$variant" == C ]]; then
    run_dir="$project_dir/outputs/replogle_llm_control_evaluation/c-observed-depth-ckpt11000-n8-v1/$run_id"
fi
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
baseline_art="$project_dir/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
export PATH="/opt/sc-evae/bin:$PATH"
export PYTHONPATH="$baseline_art/environment/python_packages:$baseline_art/source_overlay:$project_dir/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export HF_HOME="$run_dir/hf_cache" TRITON_CACHE_DIR="$run_dir/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$run_dir/inductor_cache" XDG_CACHE_HOME="$run_dir/xdg_cache"
cleanup() {
    code=$?
    printf '%s\t%s\n' "$(date -Iseconds)" "$code" >> "$run_dir/worker_exit.tsv"
    if [[ "$(id -u)" == 0 ]]; then chown -Rh "$(stat -c '%u:%g' "$project_dir")" "$run_dir"; fi
}
trap cleanup EXIT
python "${eval_args[@]}" --output "$run_dir" --batch-size "$batch_size" \
    2>&1 | tee "$run_dir/evaluation.log"
