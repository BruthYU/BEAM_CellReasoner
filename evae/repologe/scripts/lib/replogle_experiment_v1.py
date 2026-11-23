"""Render and submit isolated Replogle experiments using the author's train/eval code."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "registry.h.pjlab.org.cn/ailab-beam-beam_gpu/aivc-celltok@sha256:5bcc353a9612286f6d2ee0567f06bfbff1b8094e258177472e375c1458af02e4"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"Refusing to overwrite a different experiment file: {path}")
    if not path.exists():
        path.write_text(text)


def load_spec(path):
    return json.loads((ROOT / path).read_text())


def run_worker(spec_path):
    spec = load_spec(spec_path)
    artifacts = ROOT / spec["artifacts"]
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", WANDB_MODE="offline",
               WANDB_CACHE_DIR=str(artifacts / "wandb-cache"), TZ="Asia/Hong_Kong",
               OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
    env["PATH"] = "/opt/sc-evae/bin:" + env["PATH"]
    env["PYTHONPATH"] = str(artifacts / "source_overlay") + os.pathsep + env.get("PYTHONPATH", "")
    lock = (artifacts / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    phase, rc = "source-validation", 1
    try:
        for name, digest in load_spec(spec["artifacts"] + "/launch_sources.json").items():
            _bootstrap.verify_input(name, digest)
        for phase, command in spec["plan"]:
            command = [_bootstrap.relocate_argument(v) for v in command]
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            (artifacts / "worker_phase.tsv").write_text(f"{stamp}\t{phase}\n")
            print(stamp, phase, shlex.join(command), flush=True)
            with (artifacts / f"{phase}.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                while chunk := process.stdout.read1(64 * 1024):
                    log.write(chunk)
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                code = process.wait()
            if code:
                raise RuntimeError(f"{phase} failed (exit {code}); see {artifacts / (phase + '.log')}")
        assert (ROOT / spec["evaluation"]).is_file(), "Evaluation finished without a summary file"
        (artifacts / "completion.json").write_text(json.dumps({"status": "complete",
            "experiment": spec["experiment"], "stage1_run": spec["stage1_run"],
            "stage2_run": spec["stage2_run"], "evaluation": spec["evaluation"],
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2) + "\n")
        (artifacts / "worker_phase.tsv").write_text(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\tcomplete\n")
        phase, rc = "complete", 0
    finally:
        with (artifacts / "worker_exit.tsv").open("a") as handle:
            handle.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\t{phase}\t{rc}\n")
        if os.getuid() == 0:
            owner = ROOT.stat()
            subprocess.run(["chown", "-Rh", f"{owner.st_uid}:{owner.st_gid}", str(artifacts),
                            str(ROOT / spec["output_root"])], check=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--prior", type=str.lower, choices=["ar"], default="ar")
    p.add_argument("--quantizer", type=str.lower, choices=["fsq"], default="fsq")
    p.add_argument("--head", type=str.lower, choices=["mse", "nb"], default="nb")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-latent-tokens", type=int, default=64)
    p.add_argument("--codebook-size", type=int, default=512)
    p.add_argument("--fsq-levels", help="Optional comma-separated levels; product must equal --codebook-size")
    p.add_argument("--stage1-seed", type=int, default=401)
    p.add_argument("--stage2-seed", type=int, default=420)
    p.add_argument("--stage1-steps", type=int, default=75000)
    p.add_argument("--stage2-steps", type=int, default=75000)
    p.add_argument("--stage1-lr", type=float, default=6e-4)
    p.add_argument("--stage2-lr", type=float, default=2e-4)
    p.add_argument("--stage1-warmup", type=int, default=2500)
    p.add_argument("--stage2-warmup", type=int, default=3000)
    p.add_argument("--stage1-grad-clip", type=float, default=5.0)
    p.add_argument("--stage2-grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--eval-freq", type=int, default=5000)
    p.add_argument("--save-freq", type=int, default=5000)
    p.add_argument("--ar-d-model", type=int, default=384)
    p.add_argument("--ar-layers", type=int, default=6)
    p.add_argument("--ar-heads", type=int, default=8)
    p.add_argument("--ar-ff-dim", type=int, default=1536)
    p.add_argument("--nb-temperature", type=float, default=1.0)
    p.add_argument("--nb-min-theta", type=float, default=1e-2)
    p.add_argument("--vae-path", help="Reuse an already trained matching VAE instead of training Stage 1")
    p.add_argument("--tag", default="", help="Optional experiment label, e.g. repeat2")
    p.add_argument("--charged-group", default="beam_gpu_pool")
    p.add_argument("--gpus", type=int, choices=[1], default=1)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true")
    action.add_argument("--spec", action="store_true", help="Render rjob YAML without submitting")
    action.add_argument("--dry-run", action="store_true", help="Default: write isolated configs and print the command")
    action.add_argument("--preflight", action="store_true", help="Render and run a bounded CPU model check without submitting")
    args = p.parse_args()
    if args.worker:
        run_worker(args.worker)
        return
    for key in ["batch_size", "num_latent_tokens", "codebook_size", "stage1_steps", "stage2_steps",
                "eval_freq", "save_freq", "ar_d_model", "ar_layers", "ar_heads", "ar_ff_dim"]:
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.ar_d_model % args.ar_heads:
        p.error("--ar-d-model must be divisible by --ar-heads")
    if args.stage1_lr <= 0 or args.stage2_lr <= 0 or args.nb_temperature <= 0 or args.nb_min_theta <= 0:
        p.error("Learning rates and NB parameters must be positive")
    if args.tag and not re.fullmatch(r"[a-z0-9-]+", args.tag):
        p.error("--tag must contain lowercase letters, digits and hyphens")

    # Read author presets; adapt only paths and explicitly selected hyperparameters.
    template1 = ROOT / f"config/vaes/reploall-vae-fsq-{args.head}.yaml"
    template2 = ROOT / f"config/prior/reploall-autoregressive-fsq-{args.head}.yaml"
    s1, s2 = yaml.safe_load(template1.read_text()), yaml.safe_load(template2.read_text())
    dataset = yaml.safe_load((ROOT / "config/local/replogle-stage1-fsq-mse.yaml").read_text())["dataset"]
    embedding = load_spec("artifacts/replogle_stage2/embedding_audit.json")
    assert embedding["status"] == "passed" and embedding["feature_dim"] == 5120
    assert not embedding["heldout_missing_or_zero"]
    parameters = {k: v for k, v in vars(args).items() if k not in {"worker", "submit", "spec", "dry_run", "preflight"}}
    basis = {"config_schema": 1, "parameters": parameters, "author_stage1_sha256": sha(template1),
             "author_stage2_sha256": sha(template2), "embedding_sha256": embedding["sha256"]}
    key = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:10]
    identifier = f"ar-fsq-{args.head}-l{args.num_latent_tokens}-k{args.codebook_size}-{key}"
    if args.tag:
        identifier += "-" + args.tag
    artifact_rel = "artifacts/replogle_experiments/" + identifier
    output_rel = "outputs/replogle_experiments/" + identifier
    config_rel = "config/local/replogle_experiments/" + identifier
    artifacts = ROOT / artifact_rel
    artifacts.mkdir(parents=True, exist_ok=True)
    run1, run2 = output_rel + "/stage1", output_rel + "/stage2"
    smoke1, smoke2 = output_rel + "/stage1-smoke", output_rel + "/stage2-smoke"
    dataset.update(apply_normalize=args.head == "mse", apply_log1p=args.head == "mse",
                   use_binning="none", batch_size=args.batch_size, deg_table_path=smoke1 + "/deg_table.pt")
    s1.update(run_name="stage1", work_dir=output_rel, prefix_run_name_with_datetime=False,
              seed=args.stage1_seed, n_iters=args.stage1_steps, learning_rate=args.stage1_lr,
              warmup_steps=args.stage1_warmup, grad_clip=args.stage1_grad_clip, dataset=dict(dataset))
    s1["dataset"]["pert_mapping_path"] = None
    s1["model"].update(num_latent_tokens=args.num_latent_tokens, do_pert_cond=False, do_celltype_cond=False,
                        train_pert_classifier=False, train_pert_regressor=False)
    s1["model"]["quantizer"] = {"type": args.quantizer, "codebook_size": args.codebook_size}
    if args.fsq_levels:
        s1["model"]["quantizer"]["fsq_level"] = [int(v) for v in args.fsq_levels.split(",")]
    if args.head == "nb":
        s1["model"]["output_head"].update(temperature=args.nb_temperature, min_theta=args.nb_min_theta)
    if args.vae_path:
        run1 = str(Path(args.vae_path))
    s2.update(run_name="stage2", work_dir=output_rel, prefix_run_name_with_datetime=False,
              seed=args.stage2_seed, n_iters=args.stage2_steps, learning_rate=args.stage2_lr,
              warmup_steps=args.stage2_warmup, grad_clip=args.stage2_grad_clip, dataset=dict(dataset))
    s2["dataset"]["pert_mapping_path"] = str(Path(embedding["path"]).relative_to(ROOT))
    if args.vae_path:
        saved = yaml.safe_load((ROOT / run1 / "config.yaml").read_text())
        if saved["model"]["output_head"]["type"] != args.head:
            p.error("--vae-path output head does not match --head; an MSE tokenizer cannot stand in for NB")
        if saved["model"]["num_latent_tokens"] != args.num_latent_tokens or saved["model"]["quantizer"]["codebook_size"] != args.codebook_size:
            p.error("--vae-path token count/codebook does not match requested hyperparameters")
        s2["dataset"]["deg_table_path"] = saved["dataset"].get("deg_table_path") or str(Path(run1) / "deg_table.pt")
    s2["model"].update(vae_path=run1, do_pert_cond=True, do_celltype_cond=True, pert_cond_source="projected")
    s2["model"]["transformer"].update(d_model=args.ar_d_model, num_layers=args.ar_layers,
        nhead=args.ar_heads, dim_feedforward=args.ar_ff_dim)
    for cfg in [s1, s2]:
        cfg.update(weight_decay=args.weight_decay, mixed_precision="bf16", eval_freq=args.eval_freq,
                   save_freq=args.save_freq, log_freq=100, lr_schedule="cosine")
        cfg["wandb"] = {"enabled": True, "project": "sc_evae"}
    variants = {"stage1": s1, "stage2": s2}
    for stage, original in list(variants.items()):
        clone = json.loads(json.dumps(original))
        clone.update(run_name=stage + "-smoke", dry_run=True, profile=True,
                     profile_n_steps=50, profile_warmup_steps=10)
        clone["wandb"]["enabled"] = False
        if stage == "stage1":
            clone["dataset"]["deg_table_path"] = None
        variants[stage + "-smoke"] = clone
    from draccus import decode
    from sc_evae.config.training import TrainConfig
    for stage, cfg in variants.items():
        decoded = decode(TrainConfig, cfg)
        if stage.startswith("stage1") and decoded.model.quantizer.codebook_size != args.codebook_size:
            p.error(f"Author FSQ mapping yields {decoded.model.quantizer.codebook_size} codes; pass explicit --fsq-levels with the requested product")
        checked_write(ROOT / config_rel / (stage + ".yaml"), yaml.safe_dump(cfg, sort_keys=False))
    overlay = artifacts / "source_overlay"
    overlay.mkdir(exist_ok=True)
    if not (overlay / "sc_evae").exists():
        (overlay / "sc_evae").symlink_to(ROOT / "src/sc-evae")
    spec_rel = artifact_rel + "/experiment.json"
    validate = ["python", "scripts/checks/validate_replogle_experiment.py", "--spec", spec_rel]
    plan = [["environment", ["python", "scripts/environment/check_environment.py", "--require-cuda"]]]
    stages = [2] if args.vae_path else [1, 2]
    plan.append(["data-validation", validate + ["--phase", "data"]])
    for number in stages:
        for mode in ["smoke", "formal"]:
            suffix = "-smoke" if mode == "smoke" else ""
            config = config_rel + f"/stage{number}{suffix}.yaml"
            plan.append([f"stage{number}-{mode}-training", ["accelerate", "launch", "--num_processes", "1",
                "--num_machines", "1", "--mixed_precision", "bf16", "--dynamo_backend", "no",
                "scripts/train.py", "--config_path", config]])
            plan.append([f"stage{number}-{mode}-checkpoint", validate + ["--phase", f"stage{number}",
                         "--run", output_rel + f"/stage{number}{suffix}", "--device", "cuda"]])
    plan.append(["heldout-evaluation", ["python", "scripts/eval.py", "--train_dir", run2,
                 "--batch_size", str(args.batch_size), "--num_workers", "0", "--device", "cuda"]])
    spec = {"experiment": identifier, "parameters": parameters, "artifacts": artifact_rel,
            "output_root": output_rel, "config_root": config_rel, "stage1_run": run1, "stage2_run": run2,
            "evaluation": run2 + "/eval_results/summary_metrics.json", "plan": plan}
    checked_write(ROOT / spec_rel, json.dumps(spec, indent=2) + "\n")
    checked_write(artifacts / "parameters.json", json.dumps(basis, indent=2) + "\n")
    bundle = artifacts / "job_bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "README.txt").write_text("Shared experiment: " + spec_rel + "\n")
    job_name = f"scevae-wangyue-ar-fsq-{args.head}-{key}"
    command = ["rjob", "submit", "--name=" + job_name, "--namespace=ailab-beam", "--task_name=train",
        "--replicas=1", "--folder=" + str(bundle), "--charged-group=" + args.charged_group,
        "--private-machine=group", "--gpu=1", "--cpu=8", "--memory=64000", "--image=" + IMAGE,
        "--mount=gpfs://gpfs2/beam-gpfs02/wangyue:/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue",
        "--mount=gpfs://gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig:/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig",
        "--host-network=false", "--gang-start=false", "--auto-restart=false", "--restart-policy=never",
        "--backoff_limit=1", "--custom-resources=brainpp.cn/fuse=1"]
    if args.spec:
        command.append("--dry-run=true")
    command += ["--", "bash", str(ROOT / "scripts/launch/run_replogle_experiment.sh"), "--worker", spec_rel]
    print(json.dumps({"experiment": identifier, "job_name": job_name, "spec": spec_rel,
                      "head": args.head, "stage1": run1, "stage2": run2}, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if args.preflight or (args.submit and not (artifacts / "cpu_preflight.json").is_file()):
        env = os.environ.copy()
        env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        env["PYTHONPATH"] = str(overlay) + os.pathsep + env.get("PYTHONPATH", "")
        check = subprocess.run([sys.executable, "-u", "scripts/checks/validate_replogle_experiment.py",
            "--spec", spec_rel, "--phase", "cpu"], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (artifacts / "cpu_preflight.log").write_text(check.stdout)
        print(check.stdout, flush=True)
        check.check_returncode()
    if not (args.submit or args.spec):
        return
    if args.submit:
        receipt = artifacts / "cpu_preflight.json"
        assert receipt.is_file() and json.loads(receipt.read_text())["status"] == "passed", "Run the CPU preflight before submission"
        assert json.loads(receipt.read_text())["experiment"] == identifier
        paths = list(ROOT.glob("src/**/*.py")) + list((ROOT / config_rel).glob("*.yaml"))
        paths += [ROOT / name for name in [spec_rel, artifact_rel + "/parameters.json",
            "scripts/launch/run_replogle_experiment.sh", "scripts/lib/replogle_experiment.py",
            "scripts/checks/validate_replogle_experiment.py", "scripts/train.py", "scripts/eval.py", "scripts/environment/check_environment.py",
            "artifacts/replogle_stage1/loader_audit.json", "artifacts/replogle_stage1/data_preparation.json",
            "artifacts/replogle_stage1/raw_audit.json", "artifacts/replogle_stage1/checkpoint_sample.npz",
            "artifacts/replogle_stage2/embedding_audit.json", "assets/scldm_replogle_test.json",
            "assets/replogle_scldm2k_gene_order.ensembl.txt"]]
        paths.append(Path(embedding["path"]))
        if args.vae_path:
            paths += [ROOT / run1 / "config.yaml", ROOT / run1 / "dataset_stats.json", ROOT / run1 / "checkpoints/latest"]
            pointer = (ROOT / run1 / "checkpoints/latest").read_text().strip()
            paths.append(ROOT / run1 / "checkpoints" / pointer / "model.safetensors")
        manifest = {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))}
        checked_write(artifacts / "launch_sources.json", json.dumps(manifest, indent=2) + "\n")
        (artifacts / "submit_command.txt").write_text(shlex.join(command) + "\n")
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (artifacts / ("submit.log" if args.submit else "job_spec.yaml")).write_text(result.stdout)
    print(result.stdout, flush=True)
    result.check_returncode()
    if args.submit:
        (artifacts / "submission.json").write_text(json.dumps({"job_name": job_name, "charged_group": args.charged_group,
            "gpus": 1, "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
