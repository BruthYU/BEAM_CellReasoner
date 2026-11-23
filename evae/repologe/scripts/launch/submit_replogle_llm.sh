#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
mode="${1:---dry-run}"
case "$mode" in --dry-run|--check|--spec|--submit) ;; *) echo "Usage: $0 [--dry-run|--check|--spec|--submit]" >&2; exit 2 ;; esac
artifact_dir="$project_dir/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
job_name="${SC_EVAE_JOB_NAME:-scevae-wangyue-llm-sft-nb-2g-pool-$(date +%Y%m%d-%H%M%S)}"
image=registry.h.pjlab.org.cn/ailab-beam-beam_gpu/aivc-celltok@sha256:5bcc353a9612286f6d2ee0567f06bfbff1b8094e258177472e375c1458af02e4
args=(rjob submit "--name=$job_name" --namespace=ailab-beam --task_name=train --replicas=1
    "--folder=$artifact_dir/job_bundle" --charged-group=beam_gpu_pool --private-machine=group
    --gpu=2 "--cpu=${SC_EVAE_CPU:-16}" "--memory=${SC_EVAE_MEMORY_MIB:-98304}" "--image=$image"
    --mount=gpfs://gpfs2/beam-gpfs02/wangyue:/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue
    --mount=gpfs://gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig:/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig
    --host-network=false --gang-start=false --auto-restart=false --restart-policy=never
    --backoff_limit=1 --custom-resources=brainpp.cn/fuse=1)
[[ "$mode" != --spec ]] || args+=(--dry-run=true)
args+=(-- bash "$project_dir/scripts/workers/run_replogle_llm_worker.sh")
if [[ "$mode" == --dry-run ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit 0; fi
/usr/bin/python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import _bootstrap
from pathlib import Path
import hashlib,json
root=Path.cwd();art=root/'artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1'
for name,digest in json.loads((art/'launch_sources.json').read_text()).items():
    _bootstrap.verify_input(name, digest)
assert json.loads((art/'split.json').read_text())['validation_conditions']==200
assert json.loads((art/'cpu_preflight.json').read_text())['status']=='passed'
assert json.loads((art/'cache/completion.json').read_text())['status']=='complete'
assert not (art/'completion.json').exists(),'Pipeline already completed'
print('CPU preflight passed: internal validation frozen, NB codes cached, sources pinned')
PY
[[ "$mode" != --check ]] || exit 0
printf '%q ' "${args[@]}" > "$artifact_dir/submit_command.txt"; printf '\n' >> "$artifact_dir/submit_command.txt"
"${args[@]}" 2>&1 | tee "$artifact_dir/$job_name.submit.log"
