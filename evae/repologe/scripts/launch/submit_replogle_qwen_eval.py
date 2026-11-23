"""Submit standalone evaluation of original-text or original-control1."""
# 推荐用同名 Bash：--preset original-control1 --run-dir RUN_DIR --dry-run。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap
import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
from replogle_qwen_eval_request import ROOT, ART, OUTPUT_ROOT, PRESETS, source_config, verify_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=PRESETS, required=True)
    p.add_argument('--run-dir', type=Path, required=True, help='Completed training run or its formal directory')
    p.add_argument('--gpus', type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument('--batch-size', type=int, default=64, help='Generation batch per GPU')
    p.add_argument('--cpus', type=int, default=16)
    p.add_argument('--memory-gib', type=int, default=96)
    p.add_argument('--charged-group', default='beam_gpu')
    p.add_argument('--node', default='none')
    p.add_argument('--run-id')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--submit', action='store_true')
    args = p.parse_args()
    if min(args.batch_size, args.cpus, args.memory_gib) < 1 or args.cpus < max(2, args.gpus):
        p.error('Positive resources required; CPUs >= max(2, GPUs)')
    if (47244 + args.batch_size - 1) // args.batch_size < args.gpus:
        p.error('Batch size must give every evaluation GPU a nonempty shard')
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run_id = args.run_id or f'eval-{stamp}'
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', run_id):
        p.error('Invalid run ID')
    try:
        cfg = source_config(args.preset, args.run_dir)
    except (ValueError, OSError, KeyError) as e:
        p.error(str(e))
    os.chdir(ROOT)
    output = OUTPUT_ROOT / args.preset / run_id
    art = ART / args.preset
    request = art / 'requests' / f'{run_id}.json'
    if output.exists() or request.exists():
        p.error('Evaluation run ID already exists; choose a new --run-id')
    cfg.update(eval_gpus=args.gpus, eval_batch_size_generation=args.batch_size,
        total_cpu_threads=args.cpus, memory_gib=args.memory_gib,
        charged_group=args.charged_group, node=None if args.node == 'none' else args.node,
        output_dir=str(output))
    job = f'scevae-wy-eval-{args.preset}-{stamp}'
    command = ['rjob', 'submit', f'--name={job}', '--namespace=ailab-beam', '--task_name=eval',
        '--replicas=1', f'--folder={art / "job_bundle"}', f'--charged-group={args.charged_group}',
        '--private-machine=group', f'--gpu={args.gpus}', f'--cpu={args.cpus}',
        f'--memory={args.memory_gib * 1024}',
        '--image=registry.h.pjlab.org.cn/ailab-beam-beam_gpu/aivc-celltok@sha256:5bcc353a9612286f6d2ee0567f06bfbff1b8094e258177472e375c1458af02e4',
        '--mount=gpfs://gpfs2/beam-gpfs02/wangyue:/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue',
        '--mount=gpfs://gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig:/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig',
        '--host-network=false', '--gang-start=false', '--auto-restart=false', '--restart-policy=never',
        '--backoff_limit=1', '--custom-resources=brainpp.cn/fuse=1']
    if cfg['node']:
        command.append(f'--positive-tags=node/{cfg["node"]}')
    command += ['--', 'bash', str(ROOT / 'scripts/workers/run_replogle_qwen_eval.sh'), '--request', str(request)]
    print(json.dumps(dict(submitted=False, configuration=cfg, command=shlex.join(command)), indent=2), flush=True)
    if args.check or args.submit:
        from replogle_qwen import verify_inputs
        verify_inputs(cfg)
        cfg['source_model_sha256'] = verify_source(cfg, check_weights=True)
        print('Frozen inputs and validation-selected model verified.', flush=True)
    if not args.submit:
        return
    cfg['source_sha256'] = {str(path.relative_to(ROOT)): _bootstrap.file_hash(path)
        for path in sorted((ROOT / 'scripts').rglob('*')) if path.is_file() and path.suffix in ('.py', '.sh', '.json')}
    request.parent.mkdir(parents=True, exist_ok=True)
    with request.open('x') as f:
        json.dump(cfg, f, indent=2)
    (art / 'job_bundle').mkdir(exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True)
    (art / 'submissions').mkdir(exist_ok=True)
    (art / 'submissions' / f'{job}.json').write_text(json.dumps(dict(job=job, command=command,
        request=str(request), output=str(output), returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr), indent=2) + '\n')
    print(result.stdout + result.stderr, flush=True)
    result.check_returncode()


if __name__ == '__main__':
    main()
