#!/usr/bin/env bash
# 任务内部调用，需已有 GPU；日常提交用 scripts/launch/submit_replogle_qwen.sh。
# bash scripts/workers/run_replogle_qwen_worker.sh REQUEST_JSON train RUN_DIR
# 位置参数：提交器生成的请求JSON、smoke|benchmark|train、新运行目录。

set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
request="${1:?Request JSON required}"
stage="${2:?smoke, benchmark or train required}"
run_dir="${3:?Isolated run directory required}"
[[ "$stage" == smoke || "$stage" == benchmark || "$stage" == train ]]
baseline_art="$project_dir/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
export PATH="/opt/sc-evae/bin:$PATH"
export PYTHONPATH="$baseline_art/environment/python_packages:$baseline_art/source_overlay:$project_dir/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
read -r gpus threads auto_eval < <(python - "$request" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
print(c['gpus'], c['cpu_threads_per_process'], int(c.get('eval_after_training', False)))
PY
)
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
python - "$request" "$run_dir" <<'VERIFY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import _bootstrap
from replogle_llm_common import read_json
from replogle_qwen import verify_inputs, safe_output
cfg = read_json(sys.argv[1])
cfg.setdefault("preset", "original-control1")
safe_output(sys.argv[2], cfg)
verify_inputs(cfg)
VERIFY
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
cp "$request" "$run_dir/request.json"
export HF_HOME="$run_dir/hf_cache" TRITON_CACHE_DIR="$run_dir/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$run_dir/inductor_cache" XDG_CACHE_HOME="$run_dir/xdg_cache"
monitor_pid=""
cleanup() {
    code=$?
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
    printf '%s\t%s\n' "$(date -Iseconds)" "$code" >> "$run_dir/worker_exit.tsv"
    if [[ "$(id -u)" == 0 ]]; then chown -Rh "$(stat -c '%u:%g' "$project_dir")" "$run_dir"; fi
}
trap cleanup EXIT
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv --loop=5 > "$run_dir/gpu_telemetry.csv" 2> "$run_dir/gpu_telemetry.err" &
monitor_pid=$!

run_phase() {
    phase="$1"
    printf '%s\t%s\n' "$(date -Iseconds)" "$phase" >> "$run_dir/worker_phases.tsv"
    torchrun --standalone --nnodes=1 "--nproc-per-node=$gpus" scripts/training/train_replogle_qwen.py \
        --config "$run_dir/request.json" --phase "$phase" --output "$run_dir/$phase" \
        2>&1 | tee "$run_dir/$phase.log"
}
# Separate processes/model initialization: the formal run never starts from smoke weights.
run_phase smoke
if [[ "$stage" == benchmark ]]; then run_phase benchmark; fi
if [[ "$stage" == train ]]; then
    run_phase formal
    if [[ "$auto_eval" == 1 ]]; then
        printf '%s\tevaluation\n' "$(date -Iseconds)" >> "$run_dir/worker_phases.tsv"
        bash scripts/workers/run_replogle_qwen_eval.sh "$run_dir" 2>&1 | tee "$run_dir/evaluation.log"
    fi
fi
printf '%s\tcomplete\n' "$(date -Iseconds)" >> "$run_dir/worker_phases.tsv"
