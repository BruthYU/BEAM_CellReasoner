#!/usr/bin/env bash
# Submit a durable one-GPU job; raw matrices stay on GPFS.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
mode="${1:---dry-run}"
case "$mode" in --dry-run|--spec|--submit) ;; *) echo "Usage: $0 [--dry-run|--spec|--submit]" >&2; exit 2 ;; esac
job_name="${SC_EVAE_JOB_NAME:-scevae-wangyue-replogle-stage1-20261117}"
image=registry.h.pjlab.org.cn/ailab-beam-beam_gpu/aivc-celltok@sha256:5bcc353a9612286f6d2ee0567f06bfbff1b8094e258177472e375c1458af02e4
artifact_dir="$project_dir/artifacts/replogle_stage1"
bundle_dir="$artifact_dir/job_bundle"
mkdir -p "$bundle_dir"
printf 'Entry point: %s/scripts/workers/run_replogle_stage1_worker.sh\n' "$project_dir" > "$bundle_dir/README.txt"
args=(rjob submit "--name=$job_name" --namespace=ailab-beam --task_name=train --replicas=1
    "--folder=$bundle_dir" --charged-group=beam_gpu_pool --private-machine=group
    --gpu=1 --cpu=8 --memory=64000 "--image=$image"
    --mount=gpfs://gpfs2/beam-gpfs02/wangyue:/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue
    --mount=gpfs://gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig:/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig
    --host-network=false --gang-start=false --auto-restart=false --restart-policy=never
    --backoff_limit=1 --custom-resources=brainpp.cn/fuse=1)
[[ "$mode" != --spec ]] || args+=(--dry-run=true)
args+=(-- bash "$project_dir/scripts/workers/run_replogle_stage1_worker.sh")
if [[ "$mode" == --dry-run ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit 0; fi
if [[ "$mode" == --submit ]]; then
    /usr/bin/python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import _bootstrap
import hashlib,json,pathlib
root=pathlib.Path.cwd(); artifacts=root/'artifacts/replogle_stage1'
assert json.loads((artifacts/'raw_audit.json').read_text())['status']=='passed'
paths=list(root.glob('src/**/*.py'))+list(root.glob('config/local/replogle-stage1*.yaml'))
paths += [root/p for p in ['scripts/train.py','scripts/environment/check_environment.py',
    'scripts/workers/run_replogle_stage1_worker.sh','scripts/checks/validate_replogle_stage1.py',
    'assets/replogle_scldm2k_gene_order.ensembl.txt','assets/scldm_replogle_test.json',
    'artifacts/replogle_stage1/raw_audit.json']]
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
destination=artifacts/'launch_sources.json'
if destination.exists():
    assert json.loads(destination.read_text())==manifest, 'Previous launch manifest differs; use a separate run for changed code/config.'
else: destination.write_text(json.dumps(manifest,indent=2)+'\n')
PY
    printf '%q ' "${args[@]}" > "$artifact_dir/submit_command.txt"
    printf '\n' >> "$artifact_dir/submit_command.txt"
fi
"${args[@]}" 2>&1 | tee "$artifact_dir/submit${mode/--/.}.log"
