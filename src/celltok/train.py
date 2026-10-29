#!/usr/bin/env python3
"""Prepare CellStream loaders and call VQVAE.fit_ddp_advanced on one/N GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from lib_vqvae.advanced_training import gather_records
from lib_vqvae.cellstream_dataset import CellStreamDataset, read_archive_manifest
from lib_vqvae.factory import build_vqvae
from lib_vqvae.paths import resolve_data_path
from lib_vqvae.samplers import ArchiveBatchSampler
from celltok.data.training import LimitedLoader, audit_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tokenizer_stage1_full_primary.yaml"))
    parser.add_argument("--manifest", type=Path, help="Override both train/test manifest paths")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"])
    parser.add_argument("--scheduler", choices=["warmup_cosine", "reduce_on_plateau"])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--local-cache-dir", type=Path)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--save-every-epochs", type=int)
    parser.add_argument("--max-train-steps", type=int, help="Smoke-test step limit per epoch")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def launch_or_connect(cli):
    """Detect GPUs and launch real DDP processes, never DataParallel."""
    external_world = int(os.environ.get("WORLD_SIZE", "1"))
    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but no CUDA device is available")
    if cli.num_gpus < 1:
        raise ValueError("--num-gpus must be >= 1")
    if external_world == 1 and "LOCAL_RANK" not in os.environ and cli.device == "cuda":
        count = min(cli.num_gpus, torch.cuda.device_count())
        if count > 1:
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       f"--nproc_per_node={count}", str(Path(__file__).resolve()), *sys.argv[1:]]
            print(json.dumps({"event": "auto_torchrun", "processes": count, "requested_gpus": cli.num_gpus}), flush=True)
            subprocess.run(command, check=True)
            return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cli.device == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise ValueError("LOCAL_RANK exceeds visible CUDA devices")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if external_world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=20))
    return (dist.get_rank() if external_world > 1 else 0), external_world, device


def make_dataset(data, kind):
    return CellStreamDataset.from_manifest(
        data[f"{kind}_manifest"], split=data[f"{kind}_split"],
        expected_genes=int(data["expected_genes"]), target_sum=float(data["target_sum"]),
        log_transform=data["log_transform"], read_threads=int(data.get("cellstream_read_threads", 1)),
        max_open_archives=int(data.get("max_open_archives", 1)),
        local_cache_dir=data["local_cache_dir"],
    )


def prepare_and_audit(data, rank, world_size):
    """Only rank 0 stages/audits; other ranks reuse the locked local cache."""
    failure = [None]
    audits = []
    if rank == 0:
        try:
            studies = []
            for kind in ("train", "validation"):
                manifest = resolve_data_path(data[f"{kind}_manifest"])
                split = data[f"{kind}_split"]
                with manifest.open(newline="") as handle:
                    studies.append({r["study_id"] for r in csv.DictReader(handle) if r["split"] == split})
                sources = read_archive_manifest(manifest, split=split)
                fingerprint = {
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "genes_sha256": hashlib.sha256(Path(data["genes"]).read_bytes()).hexdigest(),
                    "split": split,
                    "sources": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources],
                }
                digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
                cache = Path(data["local_cache_dir"])/f"audit-{digest}.json"
                print(json.dumps({"event": "advanced_prepare", "split": split, "archives": len(sources)}), flush=True)
                dataset = make_dataset(data, kind)
                try:
                    if cache.is_file():
                        audit = json.loads(cache.read_text())
                        print(json.dumps({"event": "advanced_audit_cache", "split": split, "cells": len(dataset)}), flush=True)
                    else:
                        audit = audit_dataset(dataset, manifest, split, data["genes"])
                        cache.write_text(json.dumps(audit, indent=2)+"\n")
                    audits.append(audit)
                finally:
                    dataset.close()
            if studies[0] & studies[1]:
                raise ValueError("train and validation studies overlap")
        except Exception as exc:
            failure[0] = repr(exc)
    if world_size > 1:
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise ValueError(f"data preparation failed on rank 0: {failure[0]}")
    return audits


def main():
    cli = parse_args()
    context = launch_or_connect(cli)
    if context is None:
        return
    rank, world_size, device = context
    datasets = []
    try:
        torch.set_num_threads(1)
        config = yaml.safe_load(cli.config.read_text())
        data, training = config["data"], config["training"]
        if cli.manifest is not None:
            data["train_manifest"] = data["validation_manifest"] = str(cli.manifest)
        for key, value in (("learning_rate", cli.learning_rate), ("optimizer", cli.optimizer), ("scheduler", cli.scheduler)):
            if value is not None:
                training[key] = value
        if cli.workers is not None:
            if cli.workers < 0:
                raise ValueError("workers must be non-negative")
            data["num_workers_per_rank"] = cli.workers
        if cli.local_cache_dir is not None:
            data["local_cache_dir"] = str(cli.local_cache_dir)
        if data["target_sum"] != 10000 or data["log_transform"] != "log1p":
            raise ValueError("requires counts -> total-count 10000 -> natural-log log1p")
        if cli.epochs is not None:
            training["epochs"] = cli.epochs
        if cli.global_batch_size is not None:
            training["global_batch_size"] = cli.global_batch_size
        if cli.save_every_epochs is not None:
            training["save_every_epochs"] = cli.save_every_epochs
        output = cli.output_dir or Path(training["output_dir"])
        training["output_dir"] = str(output)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        existing = output/"run_manifest.json"
        if existing.exists() and cli.resume is None:
            raise FileExistsError(f"run already exists; choose a fresh --output-dir: {output}")
        batch_size = int(training.get("global_batch_size", 512))
        if batch_size < 1 or batch_size % world_size:
            raise ValueError("global batch size must be divisible by WORLD_SIZE")
        local_batch = batch_size // world_size
        seed = int(config.get("seed", 2024))
        random.seed(seed+rank)
        np.random.seed(seed+rank)
        torch.manual_seed(seed+rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed+rank)
        audits = prepare_and_audit(data, rank, world_size)
        datasets = [make_dataset(data, kind) for kind in ("train", "validation")]
        sampler = ArchiveBatchSampler(datasets[0].lengths, local_batch, seed=seed,
                                      num_replicas=world_size, rank=rank, drop_last=True)
        options = {"num_workers": int(data.get("num_workers_per_rank", 2)), "pin_memory": device.type == "cuda"}
        if options["num_workers"]:
            options["persistent_workers"] = True
        train_loader = DataLoader(datasets[0], batch_sampler=sampler, **options)
        val_sampler = DistributedSampler(datasets[1], num_replicas=world_size, rank=rank,
                                         shuffle=False, drop_last=False) if world_size > 1 else None
        eval_loader = DataLoader(datasets[1], batch_size=local_batch, sampler=val_sampler,
                                 shuffle=False, drop_last=False, **options)
        if cli.max_train_steps is not None:
            if cli.max_train_steps < 1:
                raise ValueError("smoke step limit must be positive")
            train_loader = LimitedLoader(train_loader, cli.max_train_steps)
        first_indices = next(iter(sampler), [])
        assignment = gather_records({"rank": rank, "device": str(device), "pid": os.getpid(),
                                      "first_batch_indices": first_indices,
                                      "steps": len(train_loader), "per_device_batch_size": local_batch})
        index_union = [i for record in assignment for i in record["first_batch_indices"]]
        if len(index_union) != batch_size or len(set(index_union)) != batch_size:
            raise AssertionError("first global batch has duplicate/missing samples across ranks")
        if len({r["steps"] for r in assignment}) != 1:
            raise AssertionError("rank step counts differ")
        model = build_vqvae(config).to(device)
        initial = model.encoder.network[0].weight.detach().cpu().clone() if rank == 0 else None
        fit_options = {
            "max_iter": int(training.get("epochs", 50)),
            "lr": float(training.get("learning_rate", 3e-4)),
            "weight_decay": float(training.get("weight_decay", 5e-4)),
            "log_val": int(training.get("save_every_epochs", 5)),
            "global_batch_size": batch_size,
            "warmup_ratio": float(training.get("warmup_ratio", 0.03)),
            "min_lr_ratio": float(training.get("min_lr_ratio", 0.1)),
            "eval_every_steps": int(training.get("eval_every_steps", 100)),
            "gradient_clip_norm": training.get("gradient_clip_norm"),
            "resume": cli.resume,
            "optimizer_name": training.get("optimizer", "adamw"),
            "scheduler_name": training.get("scheduler", "warmup_cosine"),
            "momentum": float(training.get("momentum", 0.9)),
            "plateau_factor": float(training.get("plateau_factor", 0.5)),
            "plateau_patience": int(training.get("plateau_patience", 50)),
            "scheduler_epochs": training.get("scheduler_epochs"),
        }
        if rank == 0:
            run = {
                "entry_point": "VQVAE.fit_ddp_advanced", "config": config,
                "world_size": world_size, "global_batch_size": batch_size,
                "per_device_batch_size": local_batch, "train_cells": len(datasets[0]),
                "validation_cells": len(datasets[1]), "steps_per_epoch": len(train_loader),
                "fit_options": {k: str(v) if isinstance(v, Path) else v for k,v in fit_options.items()},
                "rank_assignment": assignment, "first_global_batch_unique_cells": len(set(index_union)),
                "audits": audits, "created_at": datetime.now().astimezone().isoformat(),
            }
            existing.write_text(json.dumps(run, indent=2)+"\n")
            (output/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            print(json.dumps({"event": "advanced_loader_ready", "ranks": assignment,
                              "global_batch_size": batch_size, "unique_cells": len(set(index_union))}), flush=True)
        args = SimpleNamespace(outdir=str(output), config=config)
        results = model.fit_ddp_advanced(args, train_loader, None, eval_loader,
                                         config["model"]["mode"], device, **fit_options)
        failure = [None]
        if rank == 0:
            try:
                changed = not torch.equal(initial, model.encoder.network[0].weight.detach().cpu())
                if not changed or not np.isfinite(results["train_loss"]).all():
                    raise AssertionError("weights were not updated or training loss is non-finite")
                checkpoint_path = output/f"epoch_{fit_options['max_iter']}"/"model.pth"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                restored = build_vqvae(config)
                restored.load_state_dict(checkpoint["model"], strict=True)
                expected_steps = fit_options["max_iter"]*len(train_loader)
                expected_scheduler_steps = fit_options['max_iter'] if fit_options["scheduler_name"] == "reduce_on_plateau" else expected_steps
                if checkpoint["global_step"] != expected_steps or checkpoint["scheduler"]["last_epoch"] != expected_scheduler_steps:
                    raise AssertionError("checkpoint optimizer/scheduler global step mismatch")
                summary = {"event": "advanced_fit_complete", "world_size": world_size,
                           "weights_updated": changed, "checkpoint_reloaded": True,
                           "global_batch_size": batch_size, "per_device_batch_size": local_batch,
                           "global_step": checkpoint["global_step"], "checkpoint": str(checkpoint_path),
                           "optimizer": checkpoint["advanced_settings"]["optimizer"],
                           "scheduler": checkpoint["advanced_settings"]["scheduler"],
                           "scheduler_steps": checkpoint["scheduler"]["last_epoch"],
                           "parameter_sync_max_abs_diff": max(r["parameter_sync_max_abs_diff"] for r in results["history"]),
                           "final_train_loss": results["train_loss"][-1], "final_validation_loss": results["eval_loss"][-1]}
                (output/"result.json").write_text(json.dumps(summary, indent=2)+"\n")
                print(json.dumps(summary), flush=True)
            except Exception as exc:
                failure[0] = repr(exc)
        if world_size > 1:
            dist.broadcast_object_list(failure, src=0)
        if failure[0]:
            raise AssertionError(f"post-training check failed: {failure[0]}")
    finally:
        for dataset in datasets:
            dataset.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
