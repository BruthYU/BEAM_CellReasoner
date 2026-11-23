#!/usr/bin/env bash
# 在已有 GPU 的任务内评测/恢复评测，不会重新训练：
# bash scripts/workers/run_replogle_qwen_eval.sh RUN_DIR
# RUN_DIR 是含 request.json 和 formal/ 的训练运行目录，结果在其 evaluation/ 下。
# 自动运行：训练提交时加 --eval-after-training。

# Runs inside the existing training rjob after all training ranks have exited.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
entry="${1:?Run directory or --request required}"
baseline_art="$project_dir/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
export PATH="/opt/sc-evae/bin:$PATH"
export PYTHONPATH="$baseline_art/environment/python_packages:$baseline_art/source_overlay:$project_dir/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ "$entry" == --request ]]; then
    # Standalone rjob: copy the evaluation request into a new, isolated output directory.
    run_dir="$(python - "${2:?Request JSON required}" <<'INIT'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import _bootstrap
from replogle_qwen_eval_request import initialize
print(initialize(sys.argv[1]))
INIT
)"
    export HF_HOME="$run_dir/hf_cache" TRITON_CACHE_DIR="$run_dir/triton_cache"
    export TORCHINDUCTOR_CACHE_DIR="$run_dir/inductor_cache" XDG_CACHE_HOME="$run_dir/xdg_cache"
else
    run_dir="$entry"
fi
read -r eval_gpus eval_batch total_cpus < <(python - "$run_dir/request.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
print(c['eval_gpus'], c['eval_batch_size_generation'], c['total_cpu_threads'])
PY
)
pids=()
cleanup_metrics() {
    code=$?
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    if [[ "$entry" == --request ]]; then
        printf '%s\t%s\n' "$(date -Iseconds)" "$code" >> "$run_dir/worker_exit.tsv"
        if [[ "$(id -u)" == 0 ]]; then chown -Rh "$(stat -c '%u:%g' "$project_dir")" "$run_dir"; fi
    fi
}
trap cleanup_metrics EXIT
eval_dir="$run_dir/evaluation"
mkdir -p "$eval_dir"
export OMP_NUM_THREADS="$((total_cpus / eval_gpus))"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
python scripts/evaluation/eval_replogle_qwen.py prepare --run-dir "$run_dir"
# Independent replicas, no DDP metric reduction; each owns distinct global batches.
torchrun --standalone --nnodes=1 "--nproc-per-node=$eval_gpus" \
    scripts/evaluation/eval_replogle_qwen.py generate --run-dir "$run_dir" --batch-size "$eval_batch"
python scripts/evaluation/eval_replogle_qwen.py merge --run-dir "$run_dir"
# Original metrics operate on complete populations. Parallelize only independent reports.
metric_threads="$((total_cpus / 2))"
((metric_threads >= 1)) || metric_threads=1
export OMP_NUM_THREADS="$metric_threads" OPENBLAS_NUM_THREADS="$metric_threads" MKL_NUM_THREADS="$metric_threads"
mapfile -t cases < <(python - "$eval_dir/plan.json" <<'CASES'
import json,sys
print("\n".join(c["key"] for c in json.load(open(sys.argv[1]))["cases"]))
CASES
)
for split in "${cases[@]}"; do
    python scripts/evaluation/eval_replogle_qwen.py metrics --run-dir "$run_dir" --split "$split" \
        > "$eval_dir/$split-metrics.log" 2>&1 &
    pids+=("$!")
    if ((${#pids[@]} == 2)); then
        for pid in "${pids[@]}"; do wait "$pid"; done
        pids=()
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
pids=()
python scripts/evaluation/eval_replogle_qwen.py summarize --run-dir "$run_dir"
