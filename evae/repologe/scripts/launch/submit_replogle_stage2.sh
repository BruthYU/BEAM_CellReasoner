#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
mode="${1:---dry-run}"
case "$mode" in --dry-run|--spec|--submit) ;; *) echo "Usage: $0 [--dry-run|--spec|--submit]" >&2; exit 2 ;; esac
job_name="${SC_EVAE_JOB_NAME:-scevae-wangyue-replogle-stage2-ar-20261118}"
image=registry.h.pjlab.org.cn/ailab-beam-beam_gpu/aivc-celltok@sha256:5bcc353a9612286f6d2ee0567f06bfbff1b8094e258177472e375c1458af02e4
artifact_dir="$project_dir/artifacts/replogle_stage2"
mkdir -p "$artifact_dir/job_bundle" "$artifact_dir/source_overlay"
if [[ ! -e "$artifact_dir/source_overlay/sc_evae" ]]; then
    ln -s ../../../src/sc-evae "$artifact_dir/source_overlay/sc_evae"
fi
printf 'Shared entry point: %s/scripts/workers/run_replogle_stage2_worker.sh\n' "$project_dir" > "$artifact_dir/job_bundle/README.txt"
args=(rjob submit "--name=$job_name" --namespace=ailab-beam --task_name=train --replicas=1
    "--folder=$artifact_dir/job_bundle" --charged-group=beam_gpu_pool --private-machine=group
    --gpu=1 --cpu=8 --memory=64000 "--image=$image"
    --mount=gpfs://gpfs2/beam-gpfs02/wangyue:/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue
    --mount=gpfs://gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig:/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig
    --host-network=false --gang-start=false --auto-restart=false --restart-policy=never
    --backoff_limit=1 --custom-resources=brainpp.cn/fuse=1)
[[ "$mode" != --spec ]] || args+=(--dry-run=true)
args+=(-- bash "$project_dir/scripts/workers/run_replogle_stage2_worker.sh")
if [[ "$mode" == --dry-run ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit 0; fi
if [[ "$mode" == --submit ]]; then
    /usr/bin/python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import _bootstrap
import hashlib,json,pathlib
root=pathlib.Path.cwd();a=root/'artifacts/replogle_stage2'
assert json.loads((a/'embedding_audit.json').read_text())['status']=='passed'
assert json.loads((a/'model_preflight.json').read_text())['status']=='passed'
assert json.loads((a/'feature_choice.json').read_text())['feature_dim']==5120
assert not json.loads((a/'embedding_audit.json').read_text())['heldout_missing_or_zero']
stage1=json.loads((root/'artifacts/replogle_stage1/completion.json').read_text())
assert stage1['status']=='complete'
paths=list(root.glob('src/**/*.py'))+list(root.glob('config/local/replogle-stage2*.yaml'))
paths += [root/p for p in ['scripts/train.py','scripts/eval.py','scripts/environment/check_environment.py',
    'scripts/workers/run_replogle_stage2_worker.sh','scripts/checks/validate_replogle_stage2.py',
    'assets/replogle_scldm2k_gene_order.ensembl.txt','assets/scldm_replogle_test.json',
    'datasets/competition_support_set/ESM2_replogle_symbol_aligned.pt',
    'artifacts/replogle_stage1/loader_audit.json','artifacts/replogle_stage1/checkpoint_sample.npz',
    'artifacts/replogle_stage2/embedding_audit.json',
    'artifacts/replogle_stage2/feature_choice.json',
    'outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401/config.yaml',
    'outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401/dataset_stats.json',
    'outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401/checkpoints/latest',
    'outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401-smoke/deg_table.pt']]
paths.append(pathlib.Path(stage1['checkpoint'])/'model.safetensors')
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
destination=a/'launch_sources.json'
if destination.exists():
    assert json.loads(destination.read_text())==manifest, 'Launch sources differ; preserve the existing run and use a separate run for changed settings.'
else:destination.write_text(json.dumps(manifest,indent=2)+'\n')
PY
    printf '%q ' "${args[@]}" > "$artifact_dir/submit_command.txt"
    printf '\n' >> "$artifact_dir/submit_command.txt"
fi
"${args[@]}" 2>&1 | tee "$artifact_dir/submit${mode/--/.}.log"
