"""Validate the original Replogle loader and frozen Stage 1 checkpoint handoff."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import draccus
import h5py
import numpy as np
import torch

from sc_evae.config.training import TrainConfig
from sc_evae.models.factory import load_vae_from_run
from sc_evae.training.data_loader import PerturbationDataset
from sc_evae.training.train_utils import latest_checkpoint, resolve_train_val_split

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/replogle_stage1"


def save_report(name, report):
    report["status"] = "passed"
    path = ARTIFACTS / name
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def validate_data(cfg):
    raw_audit = json.loads((ARTIFACTS / "raw_audit.json").read_text())
    assert raw_audit["status"] == "passed"
    genes = (ROOT / cfg.dataset.gene_order_path).read_text().splitlines()
    split = json.loads((ROOT / cfg.dataset.split_path).read_text())
    started = time.monotonic()
    ds = PerturbationDataset(config=cfg.dataset, logger=logging.getLogger(__name__))
    train, heldout = resolve_train_val_split(ds, cfg.dataset, cfg.seed)
    assert ds.num_genes == 2000 and list(ds._target_genes) == genes
    assert ds.unique_cell_types == ["K562", "hepg2", "jurkat", "rpe1"]
    assert len(ds) == raw_audit["total_cells_after_qc"]
    assert len(train) == raw_audit["train_cells"]
    assert len(heldout) == raw_audit["heldout_cells"]
    test_mask = np.isin(ds._cell_type_indices, [ds.cell_type_to_idx[c] for c in split["cell_type"]])
    test_mask &= np.isin(ds._pert_indices, [ds.pert_to_idx[p] for p in split["pert"]])
    np.testing.assert_array_equal(heldout._indices, np.flatnonzero(test_mask))
    np.testing.assert_array_equal(train._indices, np.flatnonzero(~test_mask))
    assert set(np.asarray(ds.unique_perts)[ds._pert_indices[heldout._indices]]) == set(split["pert"])
    for start in range(0, len(ds), 8192):
        block = ds.counts[start:start + 8192]
        assert np.isfinite(block).all() and (block >= 0).all()
    files = []
    for i, raw in enumerate(raw_audit["files_in_loader_order"]):
        path = ROOT / raw["path"]
        assert path.stat().st_size == raw["raw_size_bytes"]
        assert path.stat().st_mtime_ns == raw["raw_mtime_ns"]
        start, end = map(int, ds._file_row_offsets[i:i + 2])
        assert end - start == raw["kept_cells"]
        target = float(ds._per_file_target_sum[i])
        libraries = ds._library_size[start:end]
        expected_target = cfg.dataset.target_sum or float(np.median(libraries[libraries > 0]))
        np.testing.assert_allclose(target, expected_target)
        physical_rows = np.flatnonzero(np.load(ROOT / raw["kept_cell_mask"]))
        with h5py.File(path, "r") as source:
            var = source["var"]
            native_genes = var[var.attrs.get("_index", "_index")].asstr()[:].tolist()
            positions = {g: j for j, g in enumerate(native_genes)}
            canonical = [positions[g] for g in genes]
            kept_gene_mask = ~np.isin(native_genes, raw["filtered_out_gene_ids"])
            for local in sorted({0, (end - start) // 2, end - start - 1}):
                raw_counts = source["X"][int(physical_rows[local])].astype(np.float32)
                library = float(raw_counts[kept_gene_mask].sum())
                np.testing.assert_allclose(libraries[local], library)
                expected = np.log1p(raw_counts[canonical] * np.float32(target / library))
                np.testing.assert_allclose(ds.counts[start + local], expected, rtol=2e-6, atol=2e-6)
        files.append({"cell_type": raw["cell_type"], "cells": end - start,
                      "normalization_target_sum": target})
    rng = np.random.default_rng(cfg.seed)
    sample = np.concatenate([rng.choice(train._indices, 64, replace=False),
                             rng.choice(heldout._indices, 64, replace=False)])
    np.savez_compressed(ARTIFACTS / "checkpoint_sample.npz", counts=ds.counts[sample],
                        library_size=ds._library_size[sample],
                        pert_idx=ds._pert_indices[sample], cell_type_idx=ds._cell_type_indices[sample],
                        global_row=sample)
    save_report("loader_audit.json", {
        "total_cells": len(ds), "train_cells": len(train), "heldout_cells": len(heldout),
        "genes": ds.num_genes, "cell_type_to_idx": ds.cell_type_to_idx,
        "perturbations_including_control": ds.num_perts,
        "heldout_targets": len(split["pert"]), "files": files,
        "gene_order_sha256": hashlib.sha256((ROOT / cfg.dataset.gene_order_path).read_bytes()).hexdigest(),
        "train_indices_sha256": hashlib.sha256(train._indices.tobytes()).hexdigest(),
        "heldout_indices_sha256": hashlib.sha256(heldout._indices.tobytes()).hexdigest(),
        "normalization_before_gene_selection_checked": True,
        "wall_seconds": time.monotonic() - started,
    })


def validate_checkpoint(run_dir, device="cuda"):
    device = torch.device(device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA was requested but is unavailable"
    else:
        torch.set_num_threads(2)
    run_dir = ROOT / run_dir
    checkpoint = latest_checkpoint(str(run_dir / "checkpoints"))
    assert checkpoint is not None
    for name in ["optimizer.bin", "scheduler.bin"]:
        assert (Path(checkpoint) / name).is_file(), f"Missing training state: {name}"
    assert (run_dir / "dataset_stats.json").is_file()
    with (run_dir / "config.yaml").open() as handle:
        cfg = draccus.load(TrainConfig, handle)
    # The original trainer saves locally only when computing a fresh table.
    # A run that loads the configured shared cache need not contain a copy.
    deg_table_path = ROOT / cfg.dataset.deg_table_path if cfg.dataset.deg_table_path else None
    if deg_table_path is None or not deg_table_path.is_file():
        deg_table_path = run_dir / "deg_table.pt"
    assert deg_table_path.is_file(), f"Missing DEG cache: {deg_table_path}"
    model = load_vae_from_run(str(run_dir), device=device)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    assert model.config.num_genes == 2000 and model.config.num_latent_tokens == 64
    assert model.config.quantizer.codebook_size == 512
    assert not any([model.config.do_pert_cond, model.config.do_celltype_cond,
                    model.config.train_pert_classifier, model.config.train_pert_regressor])
    with np.load(ARTIFACTS / "checkpoint_sample.npz") as sample:
        counts = torch.from_numpy(sample["counts"]).to(device)
    all_indices, losses = [], []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for block in counts.split(16 if device.type == "cuda" else 1):
            quantized = model.quantizer(model.encode(block))
            indices = quantized.indices
            assert indices.shape == (len(block), 64)
            assert indices.dtype in (torch.int32, torch.int64)
            assert indices.min() >= 0 and indices.max() < 512
            restored_codes = model.quantizer.decode_indices(indices)
            # FSQ returns bf16 codes under autocast; index decoding uses fp32.
            torch.testing.assert_close(restored_codes.float(), quantized.codes.float())
            prediction = model.decode(restored_codes)
            assert prediction.shape == block.shape and torch.isfinite(prediction).all()
            loss, _ = model({"counts": block})
            assert torch.isfinite(loss)
            losses.append(float(loss))
            all_indices.append(indices.cpu().long())
    indices = torch.cat(all_indices)
    save_report(f"{run_dir.name}-checkpoint.json", {
        "run_dir": str(run_dir), "checkpoint": checkpoint,
        "frozen_factory_load": True, "token_shape": list(indices.shape),
        "token_min": int(indices.min()), "token_max": int(indices.max()),
        "codes_used_in_sample": int(indices.unique().numel()),
        "token_decode_roundtrip": True, "sample_reconstruction_mse": float(np.mean(losses)),
        "device": str(device), "precision": "bf16" if device.type == "cuda" else "fp32",
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deg_table_path": str(deg_table_path),
        "deg_table_sha256": hashlib.sha256(deg_table_path.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((Path(checkpoint) / "model.safetensors").read_bytes()).hexdigest(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/local/replogle-stage1-fsq-mse.yaml")
    parser.add_argument("--checkpoint-run", help="Validate a frozen checkpoint instead of loading the dataset")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Checkpoint acceptance device; CPU uses one cell at a time, without gradients")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.checkpoint_run:
        validate_checkpoint(args.checkpoint_run, args.device)
    else:
        with (ROOT / args.config).open() as handle:
            cfg = draccus.load(TrainConfig, handle)
        validate_data(cfg)


if __name__ == "__main__":
    main()
