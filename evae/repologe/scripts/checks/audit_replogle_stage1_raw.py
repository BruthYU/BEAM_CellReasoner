"""Read raw Replogle matrices in bounded chunks and audit the configured QC."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import yaml

from prepare_replogle_stage1 import ROOT, read_column


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/local/replogle-stage1-fsq-mse.yaml")
    parser.add_argument("--chunk-rows", type=int, default=1024)
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())["dataset"]
    target_genes = (ROOT / cfg["gene_order_path"]).read_text().splitlines()
    split = json.loads((ROOT / cfg["split_path"]).read_text())
    preparation = json.loads((ROOT / "artifacts/replogle_stage1/data_preparation.json").read_text())
    output_dir = ROOT / "artifacts/replogle_stage1"
    files = []
    started = time.monotonic()
    for relative, metadata in zip(cfg["h5ad_paths"], preparation["files_in_loader_order"]):
        path = ROOT / relative
        assert path.name == metadata["alias"]
        file_started = time.monotonic()
        with h5py.File(path, "r") as data:
            matrix = data["X"]
            assert isinstance(matrix, h5py.Dataset), "These source files are dense H5AD matrices"
            n_cells, n_genes = matrix.shape
            genes = data["var"][data["var"].attrs.get("_index", "_index")].asstr()[:]
            perts = read_column(data["obs"], "gene")
            kept_cells = np.zeros(n_cells, dtype=bool)
            positive_cells_per_gene = np.zeros(n_genes, dtype=np.int64)
            noninteger = 0
            progress_at = 50_000
            for start in range(0, n_cells, args.chunk_rows):
                end = min(n_cells, start + args.chunk_rows)
                block = matrix[start:end]
                if not np.isfinite(block).all() or np.any(block < 0):
                    raise ValueError(f"Non-finite or negative raw counts: {path}, rows {start}:{end}")
                noninteger += int(np.count_nonzero(block != np.floor(block)))
                mask = np.count_nonzero(block, axis=1) >= cfg["min_genes"]
                kept_cells[start:end] = mask
                positive_cells_per_gene += np.count_nonzero(block[mask], axis=0)
                if end >= progress_at or end == n_cells:
                    print(f"{metadata['cell_type']}: scanned {end:,}/{n_cells:,} cells", flush=True)
                    progress_at += 50_000
            if noninteger:
                raise ValueError(f"Expected raw integer-valued counts; found {noninteger} fractional entries in {path}")
            kept_genes = positive_cells_per_gene >= cfg["min_cells"]
            retained = set(genes[kept_genes])
            missing = sorted(set(target_genes) - retained)
            if missing:
                raise ValueError(f"Canonical genes removed by configured QC in {path}: {missing}")
            heldout = np.isin(perts, split["pert"]) & kept_cells
            if metadata["cell_type"] not in split["cell_type"]:
                heldout[:] = False
            mask_name = f"raw-kept-mask-{metadata['cell_type']}.npy"
            np.save(output_dir / mask_name, kept_cells)
            report = {
                "path": relative, "cell_type": metadata["cell_type"],
                "raw_cells": n_cells, "kept_cells": int(kept_cells.sum()),
                "raw_genes": n_genes, "kept_genes": int(kept_genes.sum()),
                "canonical_genes_after_qc": len(target_genes),
                "heldout_cells": int(heldout.sum()),
                "train_cells": int(kept_cells.sum() - heldout.sum()),
                "control_cells_after_qc": int(((perts == cfg["ctrl_label"]) & kept_cells).sum()),
                "filtered_out_gene_ids": genes[~kept_genes].tolist(),
                "kept_cell_mask": str((output_dir / mask_name).relative_to(ROOT)),
                "kept_cell_mask_sha256": hashlib.sha256(kept_cells.tobytes()).hexdigest(),
                "raw_size_bytes": path.stat().st_size,
                "raw_mtime_ns": path.stat().st_mtime_ns,
                "wall_seconds": time.monotonic() - file_started,
            }
            assert report["raw_size_bytes"] == metadata["size_bytes"]
            assert report["raw_mtime_ns"] == metadata["mtime_ns"]
            files.append(report)
            print(json.dumps(report), flush=True)
    report = {
        "status": "passed", "config": str(config_path.relative_to(ROOT)),
        "files_in_loader_order": files,
        "total_cells_after_qc": sum(f["kept_cells"] for f in files),
        "train_cells": sum(f["train_cells"] for f in files),
        "heldout_cells": sum(f["heldout_cells"] for f in files),
        "wall_seconds": time.monotonic() - started,
        "raw_files_modified": False,
    }
    destination = output_dir / "raw_audit.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps({k: v for k, v in report.items() if k != "files_in_loader_order"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
