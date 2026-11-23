"""Submit a Qwen preset with configurable resources; default is a read-only dry run."""
# 推荐从 Bash 调用（自动配置环境）：
# bash scripts/launch/submit_replogle_qwen.sh --preset original-control1 --eval-after-training --dry-run
# --dry-run 换成 --submit 提交；预设与完整参数见同名 Bash 的注释和 --help。


# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from replogle_qwen_config import (ROOT, save_json, PRESETS, OUTPUT_ROOT, REQUEST_ROOT,
                                 preset_config, check_assets)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=PRESETS, required=True)
    p.add_argument("--control-cells-nums", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--epochs", type=float)
    p.add_argument("--save-steps", type=int)
    p.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--per-device-batch-size", type=int, default=16)
    p.add_argument("--global-batch-size", type=int, default=128)
    p.add_argument("--cpus", type=int, default=48)
    p.add_argument("--memory-gib", type=int, default=195)
    p.add_argument("--node", default="none")
    p.add_argument("--charged-group", default="beam_gpu")
    p.add_argument("--stage", choices=["smoke", "benchmark", "train"], default="train")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-id")
    p.add_argument("--eval-after-training", action="store_true")
    p.add_argument("--eval-gpus", type=int, help="Default: all requested training GPUs")
    p.add_argument("--eval-batch-size", type=int, default=64, help="Generation batch per GPU, independent of training batch")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--spec", action="store_true", help="Render rjob spec without submitting")
    args = p.parse_args()
    eval_gpus = args.eval_gpus if args.eval_gpus is not None else args.gpus
    if not 1 <= eval_gpus <= args.gpus or args.eval_batch_size < 1:
        p.error("Evaluation requires 1..training-GPUs workers and a positive batch size")
    if (47244 + args.eval_batch_size - 1) // args.eval_batch_size < eval_gpus:
        p.error("Evaluation batch is too large to give every GPU a nonempty shard")
    if args.eval_after_training and args.stage != "train":
        p.error("--eval-after-training requires --stage train")
    os.chdir(ROOT)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or f"{args.stage}-g{args.gpus}-b{args.per_device_batch_size}-{stamp}"
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", run_id):
        p.error("Invalid run ID")
    try:
        cfg = preset_config(args.preset, control_cells=args.control_cells_nums, gpus=args.gpus, batch_size=args.per_device_batch_size,
            global_batch=args.global_batch_size, cpus=args.cpus, memory_gib=args.memory_gib,
            node=None if args.node == "none" else args.node, charged_group=args.charged_group,
            gradient_checkpointing=args.gradient_checkpointing)
    except ValueError as e:
        p.error(str(e))
    for key in ("learning_rate", "epochs", "save_steps"):
        value = getattr(args, key)
        if value is not None:
            if value <= 0: p.error(f"{key} must be positive")
            cfg[key] = value
    cfg.update(eval_after_training=args.eval_after_training, eval_gpus=eval_gpus,
               eval_batch_size_generation=args.eval_batch_size)
    art = REQUEST_ROOT / args.preset
    output = OUTPUT_ROOT / args.preset / run_id
    if output.exists():
        p.error(f"Refusing to reuse existing output: {output}")
    request = art / "requests" / f"{run_id}.json"
    job = f"scevae-wy-{args.preset}-{args.gpus}g-{stamp}"
    command = ["rjob", "submit", f"--name={job}", "--namespace=ailab-beam", "--task_name=train",
        "--replicas=1", f"--folder={art / 'job_bundle'}", f"--charged-group={args.charged_group}",
        "--private-machine=group", f"--gpu={args.gpus}", f"--cpu={args.cpus}",
        f"--memory={args.memory_gib * 1024}",
        "--image=registry.h.pjlab.org.cn/ailab-beam-beam_gpu/aivc-celltok@sha256:5bcc353a9612286f6d2ee0567f06bfbff1b8094e258177472e375c1458af02e4",
        "--mount=gpfs://gpfs2/beam-gpfs02/wangyue:/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue",
        "--host-network=false", "--gang-start=false", "--auto-restart=false", "--restart-policy=never",
        "--backoff_limit=1", "--custom-resources=brainpp.cn/fuse=1"]
    if args.eval_after_training:
        command.append("--mount=gpfs://gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig:/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig")
    if cfg["node"]:
        command.append(f"--positive-tags=node/{cfg['node']}")
    command += ["--", "bash", str(ROOT / "scripts/workers/run_replogle_qwen_worker.sh"),
                str(request), args.stage, str(output)]
    print(json.dumps({"submitted": False, "run_id": run_id, "resources": cfg,
                      "command": shlex.join(command)}, indent=2), flush=True)
    if args.check or args.submit or args.spec:
        try:
            check_assets(cfg)
            from replogle_qwen import verify_inputs
            verify_inputs(cfg)
        except ModuleNotFoundError as e:
            p.error(f"Missing training dependency {e.name}; install .[qwen] as described in SETUP.md")
        except (FileNotFoundError, ValueError, AssertionError) as e:
            p.error(f"{e}\nSee SETUP.md for external assets and frozen-input verification.")
        print("Frozen inputs verified.", flush=True)
    if args.spec:
        command.insert(command.index("--"), "--dry-run=true")
        subprocess.run(command, check=True)
        return
    if not args.submit:
        return
    assert not request.exists(), f"Submission already reserved: {request}"
    request.parent.mkdir(parents=True, exist_ok=True)
    # An exclusive reservation prevents concurrent submissions sharing output.
    cfg["source_sha256"] = {str(path.relative_to(ROOT)): _bootstrap.file_hash(path)
        for path in sorted((ROOT / "scripts").rglob("*")) if path.suffix in (".py", ".sh", ".json")}
    with request.open("x") as f:
        json.dump(cfg, f, indent=2)
    (art / "job_bundle").mkdir(exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True)
    save_json(art / "submissions" / f"{job}.json", {"job": job, "run_id": run_id,
        "returncode": result.returncode, "command": command, "request": str(request),
        "output": str(output), "stdout": result.stdout, "stderr": result.stderr})
    print(result.stdout + result.stderr, flush=True)
    result.check_returncode()


if __name__ == "__main__":
    main()
