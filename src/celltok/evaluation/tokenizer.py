#!/usr/bin/env python3
"""Evaluate a CellTok VQ-VAE checkpoint on CellStream archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import yaml

from lib_vqvae.cellstream_dataset import CellStreamDataset
from lib_vqvae.factory import build_vqvae
from lib_vqvae.metrics import (
    clustering_metrics,
    codebook_usage_metrics,
    rbf_mmd,
    reconstruction_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tokenizer_stage1_full_primary.yaml"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--log-transform", choices=["log1p", "log10", "none"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reconstruction-only", action="store_true", help="Stream MSE/PCC/ACS in bounded memory; omit MMD/clustering")
    parser.add_argument("--all-cells", action="store_true", help="Every row in the selected split; requires --reconstruction-only")
    parser.add_argument("--verify-reference", type=Path, help="Require reconstruction metrics and cell/checkpoint hashes to match an existing result")
    args = parser.parse_args()
    if args.all_cells and (not args.reconstruction_only or args.max_cells is not None):
        parser.error("--all-cells requires --reconstruction-only and cannot be combined with --max-cells")
    if args.verify_reference and not args.reconstruction_only:
        parser.error("--verify-reference requires --reconstruction-only")
    return args


def main() -> None:
    started = time.monotonic()
    args = parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    if args.reconstruction_only:
        from celltok.evaluation.reconstruction import evaluate_reconstruction
        result = evaluate_reconstruction(args, config, started)
        rendered = json.dumps(result, indent=2)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered+"\n")
        return
    evaluation = config["evaluation"]
    data = config["data"]
    if args.log_transform is not None:
        data["log_transform"] = args.log_transform
    manifest = args.manifest or Path(data["validation_manifest"])
    split = args.split if args.split is not None else data.get("validation_split")
    dataset = CellStreamDataset.from_manifest(
        manifest,
        split=split,
        expected_genes=int(data["expected_genes"]),
        target_sum=float(data["target_sum"]),
        log_transform=str(data["log_transform"]),
        read_threads=int(data.get("cellstream_read_threads", 1)),
        local_cache_dir=data.get("local_cache_dir"),
    )
    max_cells = args.max_cells or int(evaluation["max_cells"])
    rng = np.random.default_rng(int(evaluation["random_seed"]))
    indices = np.arange(len(dataset), dtype=np.int64)
    if indices.size > max_cells:
        indices = np.sort(rng.choice(indices, max_cells, replace=False))
    labels = dataset.get_obs_values(indices, str(evaluation["label_column"]))
    seed = int(evaluation["random_seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_vqvae(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    model.eval()

    original_batches = []
    reconstructed_batches = []
    latent_batches = []
    code_index_batches = []
    batch_size = int(evaluation["batch_size"])
    total_batches = (indices.size + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_number, start in enumerate(
            range(0, indices.size, batch_size), start=1
        ):
            batch_indices = indices[start : start + batch_size]
            original = dataset.get_batch_numpy(batch_indices)
            x = torch.from_numpy(original).to(device)
            _, _, _, x_hat, _, latent, _, _, code_indices = model(
                x, config["model"]["mode"], device
            )
            original_batches.append(original)
            reconstructed_batches.append(x_hat.cpu().numpy())
            latent_batches.append(latent.cpu().numpy())
            code_index_batches.append(
                code_indices.reshape(x.shape[0], int(config["model"]["n"]))
                .cpu()
                .numpy()
            )
            if batch_number % 100 == 0 or batch_number == total_batches:
                print(
                    f"inference batches {batch_number}/{total_batches}",
                    flush=True,
                )
    dataset.close()

    original = np.concatenate(original_batches)
    reconstructed = np.concatenate(reconstructed_batches)
    latent = np.concatenate(latent_batches)
    code_indices = np.concatenate(code_index_batches)
    # Do not retain duplicate full matrices during float64 metric computation.
    del original_batches, reconstructed_batches, latent_batches, code_index_batches
    print("computing reconstruction metrics", flush=True)
    metrics = reconstruction_metrics(original, reconstructed)
    mmd_squared, median_squared_distance = rbf_mmd(
        original,
        reconstructed,
        max_cells=int(evaluation["mmd_max_cells"]),
        bandwidth_sample=int(evaluation["mmd_bandwidth_sample"]),
        seed=int(evaluation["random_seed"]),
    )
    metrics["mmd"] = mmd_squared
    metrics["mmd_rbf_biased_squared"] = mmd_squared
    # The paper's reported 0.0658 is almost exactly half of the explicit
    # biased MMD^2 estimate on the calibrated 10k subset (0.131313 / 2).
    # Preserve both values instead of silently selecting an undocumented
    # convention.
    metrics["mmd_half_biased_squared"] = mmd_squared / 2.0
    metrics["mmd_median_squared_distance"] = median_squared_distance
    print("computing reconstruction clustering", flush=True)
    metrics["reconstruction_clustering"] = clustering_metrics(
        reconstructed,
        labels,
        pca_components=int(evaluation["pca_components"]),
        seed=int(evaluation["random_seed"]),
    )
    print("computing latent clustering", flush=True)
    metrics["latent_clustering"] = clustering_metrics(
        latent,
        labels,
        pca_components=None,
        seed=int(evaluation["random_seed"]),
    )
    metrics["codebook_usage"] = codebook_usage_metrics(
        code_indices,
        n_codes=int(config["model"]["n_codebooks"]),
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(Path(manifest).resolve()),
        "split": split,
        "cells": int(indices.size),
        "normalization": data["log_transform"],
        "protocol": {
            "seed": seed,
            "label_column": str(evaluation["label_column"]),
            "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "target_sum": float(data["target_sum"]),
            "expected_genes": int(data["expected_genes"]),
            "evaluation": evaluation,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "device": str(device),
        },
        "elapsed_seconds": time.monotonic() - started,
        "metrics": metrics,
    }
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    result["checkpoint_sha256"] = digest.hexdigest()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
