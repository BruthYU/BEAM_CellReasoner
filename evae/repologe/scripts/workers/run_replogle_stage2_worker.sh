#!/usr/bin/env bash
# Reuse the original train/eval entry points, with the completed Stage 1 VAE.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
artifact_dir="$project_dir/artifacts/replogle_stage2"
output_parent="$project_dir/outputs/replogle_stage2"
mkdir -p "$artifact_dir" "$output_parent"
export PATH="/opt/sc-evae/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export WANDB_MODE=offline WANDB_CACHE_DIR="$artifact_dir/wandb-cache" TZ=Asia/Hong_Kong
# Match the wheel's sc-evae -> sc_evae import mapping while using the checkout.
export PYTHONPATH="$artifact_dir/source_overlay${PYTHONPATH:+:$PYTHONPATH}"
exec 9>"$artifact_dir/worker.lock"
flock -n 9 || { echo 'Another Stage 2 worker is already running.' >&2; exit 2; }
exec > >(tee -a "$artifact_dir/worker.log") 2>&1
owner_uid="$(stat -c %u "$project_dir")"
owner_gid="$(stat -c %g "$project_dir")"
phase=initializing
cleanup() {
    rc=$?
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$phase" "$rc" >> "$artifact_dir/worker_exit.tsv"
    if [[ "$(id -u)" == 0 ]]; then
        chown -Rh "$owner_uid:$owner_gid" "$artifact_dir" "$output_parent" || true
    fi
    exit "$rc"
}
trap cleanup EXIT
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import _bootstrap
import hashlib,json,pathlib,sc_evae
root=pathlib.Path.cwd()
assert pathlib.Path(sc_evae.__file__).resolve()==root/'src/sc-evae/__init__.py'
for name, expected in json.loads((root/'artifacts/replogle_stage2/launch_sources.json').read_text()).items():
    _bootstrap.verify_input(name, expected)
PY
run_phase() {
    phase="$1"; shift
    printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$phase" | tee "$artifact_dir/worker_phase.tsv"
    "$@" 2>&1 | tee -a "$artifact_dir/$phase.log"
}
run_phase environment python scripts/environment/check_environment.py --require-cuda
run_phase loader-validation python scripts/checks/validate_replogle_stage2.py --data
smoke_run=outputs/replogle_stage2/replogle-stage2-ar-fsq-mse-seed420-smoke
formal_run=outputs/replogle_stage2/replogle-stage2-ar-fsq-mse-seed420
if [[ ! -f "$artifact_dir/replogle-stage2-ar-fsq-mse-seed420-smoke-checkpoint.json" ]]; then
    run_phase smoke-training accelerate launch --num_processes 1 --num_machines 1 \
        --mixed_precision bf16 --dynamo_backend no scripts/train.py \
        --config_path config/local/replogle-stage2-ar-fsq-mse-smoke.yaml
fi
run_phase smoke-checkpoint python scripts/checks/validate_replogle_stage2.py --checkpoint-run "$smoke_run" --device cuda
run_phase formal-training accelerate launch --num_processes 1 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no scripts/train.py \
    --config_path config/local/replogle-stage2-ar-fsq-mse.yaml
run_phase formal-checkpoint python scripts/checks/validate_replogle_stage2.py --checkpoint-run "$formal_run" --device cuda
# Training validation is teacher-forced; the original eval performs real sampling.
run_phase heldout-evaluation python scripts/eval.py --train_dir "$formal_run" \
    --batch_size 128 --num_workers 0 --device cuda
phase=complete
printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$phase" | tee "$artifact_dir/worker_phase.tsv"
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import _bootstrap
import datetime,json,pathlib
root=pathlib.Path.cwd();a=root/'artifacts/replogle_stage2'
run=root/'outputs/replogle_stage2/replogle-stage2-ar-fsq-mse-seed420'
check=json.loads((a/(run.name+'-checkpoint.json')).read_text())
assert check['status']=='passed'
summary=run/'eval_results/summary_metrics.json'
assert summary.is_file()
(a/'completion.json').write_text(json.dumps({'status':'complete','run_dir':str(run),
    'checkpoint':check['checkpoint'],'heldout_evaluation_summary':str(summary),
    'source_vae_weights_unchanged':True,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
