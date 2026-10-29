"""Bounded-memory reconstruction metrics and full-cohort CellStream evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
import torch
from tqdm import tqdm

from celltok.data.training import audit_dataset
from lib_vqvae.cellstream_dataset import CellStreamDataset
from lib_vqvae.factory import build_vqvae


class StreamingReconstructionMetrics:
    """Same definitions as reconstruction_metrics; retain only O(genes) state."""

    def __init__(self, genes: int):
        if genes < 1:
            raise ValueError("genes must be positive")
        self.genes = genes
        self.cells = 0
        self.squared_error = self.pcc_sum = self.cosine_sum = 0.0
        self.dot = np.zeros(genes, dtype=np.float64)
        self.original_squared = np.zeros(genes, dtype=np.float64)
        self.reconstructed_squared = np.zeros(genes, dtype=np.float64)

    def update(self, original, reconstructed):
        x = np.asarray(original, dtype=np.float64)
        y = np.asarray(reconstructed, dtype=np.float64)
        if x.ndim != 2 or x.shape != y.shape or x.shape[1] != self.genes:
            raise ValueError("expected matching 2-D batches with the configured gene count")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("non-finite expression or reconstruction")
        if not x.shape[0]:
            return
        difference = x - y
        self.squared_error += float(np.einsum("ij,ij->", difference, difference))
        self.dot += np.einsum("ij,ij->j", x, y)
        self.original_squared += np.einsum("ij,ij->j", x, x)
        self.reconstructed_squared += np.einsum("ij,ij->j", y, y)
        denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        self.cosine_sum += float(np.divide(
            np.einsum("ij,ij->i", x, y), denominator,
            out=np.zeros(x.shape[0], dtype=np.float64), where=denominator > 0,
        ).sum())
        # Explicit centering retains the reference behavior for constant rows.
        x = x - x.mean(axis=1, keepdims=True)
        y = y - y.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        self.pcc_sum += float(np.divide(
            np.einsum("ij,ij->i", x, y), denominator,
            out=np.zeros(x.shape[0], dtype=np.float64), where=denominator > 0,
        ).sum())
        self.cells += x.shape[0]

    def result(self):
        if not self.cells:
            raise ValueError("no cells evaluated")
        denominator = np.sqrt(self.original_squared) * np.sqrt(self.reconstructed_squared)
        gene_cosine = np.divide(self.dot, denominator, out=np.zeros(self.genes), where=denominator > 0)
        global_denominator = np.sqrt(self.original_squared.sum()) * np.sqrt(self.reconstructed_squared.sum())
        return {
            "mse": self.squared_error / (self.cells * self.genes),
            "pcc": self.pcc_sum / self.cells,
            "acs": self.cosine_sum / self.cells,
            "acs_cellwise": self.cosine_sum / self.cells,
            "acs_genewise": float(gene_cosine.mean()),
            "acs_global": float(self.dot.sum() / global_denominator) if global_denominator > 0 else 0.0,
        }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_reconstruction(args, config, started):
    """Evaluate every selected row without retaining expression/latent matrices."""
    data = config["data"]
    if args.log_transform is not None:
        data["log_transform"] = args.log_transform
    if data["log_transform"] != "log1p" or float(data["target_sum"]) != 10000:
        raise ValueError("Primary archive audit requires log1p and target_sum=10000")
    manifest = args.manifest or Path(data["validation_manifest"])
    split = args.split if args.split is not None else data.get("validation_split", "test")
    settings = config.get("evaluation", {})
    seed = int(settings.get("random_seed", config.get("seed", 2024)))
    batch_size = int(settings.get("batch_size", 256))
    genes = int(data["expected_genes"])
    if batch_size < 1:
        raise ValueError("evaluation batch_size must be positive")
    with Path(manifest).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [r for r in rows if r["split"] == split]
    training_studies = {r.get("study_id") for r in rows if r["split"] == "train"}
    selected_studies = {r.get("study_id") for r in selected}
    if split != "train" and training_studies & selected_studies:
        raise ValueError("evaluation and training study overlap")
    dataset = CellStreamDataset.from_manifest(
        manifest, split=split, expected_genes=genes, target_sum=float(data["target_sum"]),
        log_transform=data["log_transform"], read_threads=int(data.get("cellstream_read_threads", 1)),
        max_open_archives=int(data.get("max_open_archives", 1)), local_cache_dir=data.get("local_cache_dir"),
    )
    try:
        audit = audit_dataset(dataset, manifest, split, data.get("genes", "data/5000_gene_names.csv"))
        if args.all_cells:
            indices = None
            cells = len(dataset)
        else:
            cells = min(args.max_cells or int(settings.get("max_cells", 10000)), len(dataset))
            if cells < 1:
                raise ValueError("max_cells must be positive")
            indices = (np.sort(np.random.default_rng(seed).choice(len(dataset), cells, replace=False))
                       if cells < len(dataset) else None)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        model = build_vqvae(config).to(device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        model.eval()
        accumulator = StreamingReconstructionMetrics(genes)
        index_digest = hashlib.sha256()
        inference_started = time.monotonic()
        progress_path = args.output.with_suffix(".progress.json") if args.output else None
        if progress_path:
            progress_path.parent.mkdir(parents=True, exist_ok=True)

        def progress(status):
            elapsed = time.monotonic() - inference_started
            speed = accumulator.cells / max(elapsed, 1e-9)
            value = {"event": "reconstruction_progress", "status": status,
                     "checkpoint": str(args.checkpoint), "cells_done": accumulator.cells,
                     "total_cells": cells, "elapsed_seconds": elapsed,
                     "cells_per_second": speed, "eta_seconds": (cells-accumulator.cells)/speed if speed else None,
                     "updated_at": datetime.now().astimezone().isoformat()}
            print(json.dumps(value), flush=True)
            if progress_path:
                temporary = progress_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(value, indent=2)+"\n")
                temporary.replace(progress_path)

        progress("running")
        with torch.inference_mode(), tqdm(total=cells, desc="Reconstruction", unit="cell", mininterval=5) as bar:
            for batch, start in enumerate(range(0, cells, batch_size), 1):
                stop = min(start+batch_size, cells)
                batch_indices = np.arange(start, stop, dtype=np.int64) if indices is None else indices[start:stop]
                original = dataset.get_batch_numpy(batch_indices)
                outputs = model(torch.from_numpy(original).to(device), config["model"]["mode"], device)
                accumulator.update(original, outputs[3].cpu().numpy())
                index_digest.update(batch_indices.tobytes())
                bar.update(stop-start)
                if batch % 100 == 0 or stop == cells:
                    progress("running")
        if accumulator.cells != cells:
            raise RuntimeError("incomplete evaluation")
        result = {
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
            "manifest": str(Path(manifest).resolve()), "split": split, "cells": cells,
            "normalization": data["log_transform"], "elapsed_seconds": time.monotonic()-started,
            "protocol": {
                "scope": "full_split_reconstruction" if args.all_cells else "sampled_reconstruction",
                "seed": seed, "indices_sha256": index_digest.hexdigest(),
                "manifest_sha256": file_sha256(manifest), "gene_vocabulary_sha256": audit["gene_vocabulary_sha256"],
                "target_sum": float(data["target_sum"]), "expected_genes": genes,
                "evaluation": {"batch_size": batch_size, "all_cells": args.all_cells, "selected_cells": cells,
                               "accumulator_dtype": "float64", "undefined_cell_correlations": "zero",
                               "mse_reduction": "sum_squared_error/(cells*genes)",
                               "pcc_reduction": "mean of per-cell Pearson coefficients",
                               "acs_genewise_reduction": "whole-cohort dot products and squared norms per gene",
                               "mmd_and_clustering": "not computed; see separate fixed-cohort evaluations"},
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "sklearn_version": sklearn.__version__, "device": str(device),
            },
            "data_audit": audit, "metrics": accumulator.result(),
        }
        if args.verify_reference:
            reference = json.loads(args.verify_reference.read_text())
            for key in ("cells", "normalization", "split", "manifest", "checkpoint_sha256"):
                if result[key] != reference[key]:
                    raise ValueError(f"reference mismatch: {key}")
            for key in ("indices_sha256", "manifest_sha256", "target_sum", "expected_genes"):
                if result["protocol"][key] != reference["protocol"][key]:
                    raise ValueError(f"reference protocol mismatch: {key}")
            errors = {}
            for key, value in result["metrics"].items():
                np.testing.assert_allclose(value, reference["metrics"][key], rtol=2e-6, atol=2e-7,
                                           err_msg=f"streaming reference metric: {key}")
                errors[key] = abs(value-reference["metrics"][key])
            result["reference_validation"] = {"path": str(args.verify_reference), "passed": True,
                                                "absolute_errors": errors, "rtol": 2e-6, "atol": 2e-7}
        progress("complete")
        return result
    finally:
        dataset.close()
