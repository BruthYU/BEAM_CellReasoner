"""Submit one independent A, B or C evaluation using its frozen checkpoint/protocol."""
# 推荐从 Bash 调用；--variant 可选 A、B、C：
# bash scripts/launch/submit_replogle_control_eval.sh --variant B --dry-run
# --dry-run 换成 --submit 提交；全部参数见 --help。


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
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant_positional", nargs="?", choices=["A", "B", "C"], help="Legacy positional alias")
    p.add_argument("--variant", choices=["A", "B", "C"], help="Evaluate exactly one experiment")
    p.add_argument("--dry-run", action="store_true", help="Print only (also the default)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--node", default="gpu-lg-cmc-h-h200-0863.host.h.pjlab.org.cn")
    args = p.parse_args()
    if args.variant and args.variant_positional and args.variant != args.variant_positional:
        p.error("Conflicting variants")
    args.variant = args.variant or args.variant_positional
    if not args.variant: p.error("--variant A|B|C is required")
    if args.batch_size < 1: p.error("--batch-size must be positive")
    if args.dry_run and args.submit: p.error("--dry-run and --submit are mutually exclusive")
    root = Path(__file__).resolve().parents[2]; os.chdir(root)
    art = root / "artifacts/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-v1/evaluation_observed_depth_v1"
    if args.variant == "C":
        art = root / "artifacts/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-c-v1/evaluation_observed_depth_ckpt11000_n8_v1"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job = f"scevae-wangyue-eval-control-{args.variant.lower()}-{stamp}"
    run_id = f"eval-seed420-{stamp}"
    command = ["rjob", "submit", f"--name={job}", "--namespace=ailab-beam", "--task_name=eval", "--replicas=1",
        f"--folder={art / 'job_bundle'}", "--charged-group=beam_gpu", "--private-machine=group",
        "--gpu=1", "--cpu=16", "--memory=98304",
        "--image=registry.h.pjlab.org.cn/ailab-beam-beam_gpu/aivc-celltok@sha256:5bcc353a9612286f6d2ee0567f06bfbff1b8094e258177472e375c1458af02e4",
        "--mount=gpfs://gpfs2/beam-gpfs02/wangyue:/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue",
        "--mount=gpfs://gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig:/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig",
        "--host-network=false", "--gang-start=false", "--auto-restart=false", "--restart-policy=never",
        "--backoff_limit=1", "--custom-resources=brainpp.cn/fuse=1"]
    if args.node != "none": command.append(f"--positive-tags=node/{args.node}")
    command += ["--", "bash", str(root / "scripts/workers/run_replogle_control_eval_worker.sh"), args.variant, run_id, str(args.batch_size)]
    print(shlex.join(command), flush=True)
    if not args.submit: return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
        PYTHONPATH=str(root / "artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1/environment/python_packages")+":"+str(root / "artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1/source_overlay")+":"+str(root / "scripts"))
    subprocess.run(["/mnt/shared-storage-user/wangyue/conda/envs/sc-evae/bin/python",
                    "scripts/data/prepare_replogle_control_c_eval.py" if args.variant == "C" else "scripts/checks/check_replogle_control_eval.py", "--verify-only"], env=env, check=True)
    (art / "submissions").mkdir(parents=True, exist_ok=True)
    (art / "job_bundle").mkdir(exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True)
    (art / "submissions" / f"{job}.log").write_text(result.stdout+result.stderr)
    print(result.stdout+result.stderr, flush=True); result.check_returncode()
    data = json.loads(subprocess.check_output(["brainctl", "get", "rjob", job, "-n", "ailab-beam", "-o", "json"]))
    (art / "submissions" / f"{job}.json").write_text(json.dumps(data, indent=2)+"\n")
    path = art / "active_submissions.json"
    records = json.loads(path.read_text()) if path.exists() else []
    records.append(dict(variant=args.variant, job=job, run_id=run_id, status=data["status"]["phase"],
        output=str((root / "outputs/replogle_llm_control_evaluation/c-observed-depth-ckpt11000-n8-v1" if args.variant == "C" else root / "outputs/replogle_llm_control_evaluation/ab-observed-depth-v1" / args.variant) / run_id),
        node_constraint=args.node, charged_group="beam_gpu", gpu=1, cpu=16, memory_gib=96))
    path.write_text(json.dumps(records, indent=2)+"\n")
    print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    main()
