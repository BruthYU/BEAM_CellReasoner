#!/usr/bin/env bash
# Execute the original training entry point after data and checkpoint acceptance.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export PATH="/opt/sc-evae/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export WANDB_MODE=offline WANDB_CACHE_DIR="$project_dir/artifacts/replogle_stage1/wandb-cache"
export TZ=Asia/Hong_Kong
artifact_dir="$project_dir/artifacts/replogle_stage1"
mkdir -p "$artifact_dir"
exec 9>"$artifact_dir/worker.lock"
flock -n 9 || { echo 'A Replogle Stage 1 worker is already running.' >&2; exit 2; }
exec > >(tee -a "$artifact_dir/worker.log") 2>&1
phase=initializing
trap 'rc=$?; printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$phase" "$rc" >> "$artifact_dir/worker_exit.tsv"' EXIT
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import _bootstrap
import hashlib,json,pathlib
root=pathlib.Path.cwd()
manifest=json.loads((root/'artifacts/replogle_stage1/launch_sources.json').read_text())
for name, expected in manifest.items():
    _bootstrap.verify_input(name, expected)
assert json.loads((root/'artifacts/replogle_stage1/raw_audit.json').read_text())['status']=='passed'
PY
run_phase() {
    phase="$1"; shift
    printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$phase" | tee "$artifact_dir/worker_phase.tsv"
    "$@" 2>&1 | tee -a "$artifact_dir/$phase.log"
}
run_phase environment python scripts/environment/check_environment.py --require-cuda
run_phase loader-validation python scripts/checks/validate_replogle_stage1.py
smoke_run=outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401-smoke
formal_run=outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401
if [[ ! -f "$artifact_dir/replogle-stage1-fsq-mse-seed401-smoke-checkpoint.json" ]]; then
    run_phase smoke-training accelerate launch --num_processes 1 --num_machines 1 \
        --mixed_precision bf16 --dynamo_backend no scripts/train.py \
        --config_path config/local/replogle-stage1-fsq-mse-smoke.yaml
fi
run_phase smoke-checkpoint python scripts/checks/validate_replogle_stage1.py --checkpoint-run "$smoke_run"
run_phase formal-training accelerate launch --num_processes 1 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no scripts/train.py \
    --config_path config/local/replogle-stage1-fsq-mse.yaml
run_phase formal-checkpoint python scripts/checks/validate_replogle_stage1.py --checkpoint-run "$formal_run"
phase=complete
printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$phase" | tee "$artifact_dir/worker_phase.tsv"
