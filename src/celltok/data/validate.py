#!/usr/bin/env python3
"""Validate every CellStream archive in a manifest on an mmap-capable disk."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from lib_vqvae.paths import resolve_data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--genes", type=Path, default=Path("data/5000_gene_names.csv")
    )
    parser.add_argument("--local-cache-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument(
        "--keep-staged",
        action="store_true",
        help="Keep validated files under the loader's deterministic cache names",
    )
    return parser.parse_args()


def expected_cells(row: dict[str, str]) -> int:
    for column in (
        "selected_cell_count",
        "primary_cell_count",
        "dataset_total_cell_count",
    ):
        value = row.get(column, "")
        if value:
            return int(float(value))
    raise ValueError(
        "manifest has no selected_cell_count/primary_cell_count/dataset_total_cell_count"
    )


def resolve_archive(manifest: Path, row: dict[str, str]) -> Path:
    value = row.get("cellstream_path") or row.get("path")
    if not value:
        raise ValueError("manifest row has no cellstream_path/path")
    return resolve_data_path(value, base_dir=resolve_data_path(manifest).parent)


def read_target_genes(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "gene" not in rows[0]:
        raise ValueError(f"{path} must contain a non-empty 'gene' column")
    genes = [row["gene"] for row in rows]
    if len(genes) != len(set(genes)):
        raise ValueError(f"target gene vocabulary contains duplicates: {path}")
    return genes


def main() -> None:
    args = parse_args()
    if args.sample_rows < 0:
        raise ValueError("--sample-rows must be non-negative")
    manifest = args.manifest.resolve()
    genes_path = args.genes.resolve()
    target_genes = read_target_genes(genes_path)
    vocabulary_sha256 = hashlib.sha256(genes_path.read_bytes()).hexdigest()
    cache_dir = args.local_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.split:
        selected = set(args.split)
        rows = [row for row in rows if row.get("split") in selected]
    if not rows:
        raise ValueError("no manifest rows selected")

    import cellstream

    if args.keep_staged:
        required = 0
        for row in rows:
            source = resolve_archive(manifest, row).resolve()
            prefix = hashlib.sha256(str(source).encode()).hexdigest()[:12]
            destination = cache_dir / f"{prefix}-{source.name}"
            if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
                required += source.stat().st_size
        free = shutil.disk_usage(cache_dir).free
        print(f"staging preflight required={required} free={free}", flush=True)
        if required > free:
            raise OSError(
                f"CellStream local cache needs {required} bytes but {cache_dir} has {free} free"
            )

    report_rows: list[dict[str, object]] = []
    failures = 0
    total_cells = 0
    total_bytes = 0
    for position, row in enumerate(rows, start=1):
        dataset_id = row.get("dataset_id", "")
        source = resolve_archive(manifest, row).resolve()
        expected = expected_cells(row)
        result: dict[str, object] = {
            "position": position,
            "dataset_id": dataset_id,
            "split": row.get("split", ""),
            "archive": str(source),
            "expected_cells": expected,
            "actual_cells": "",
            "actual_genes": "",
            "gene_order_matches": "",
            "archive_vocabulary_sha256": "",
            "bytes": "",
            "sha256": "",
            "status": "failed",
            "error": "",
        }
        temporary: Path | None = None
        keep_temporary = False
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            size = source.stat().st_size
            total_bytes += size
            result["bytes"] = size
            if args.keep_staged:
                prefix = hashlib.sha256(str(source).encode()).hexdigest()[:12]
                destination = cache_dir / f"{prefix}-{source.name}"
                temporary = destination.with_suffix(destination.suffix + ".validation-partial")
                if destination.is_file() and destination.stat().st_size == size:
                    temporary = destination
                    keep_temporary = True
            else:
                with tempfile.NamedTemporaryFile(
                    prefix=f"{dataset_id}-", suffix=".csad", dir=cache_dir, delete=False
                ) as handle:
                    temporary = Path(handle.name)
            digest = hashlib.sha256()
            if not keep_temporary:
                with source.open("rb") as source_handle, temporary.open("wb") as output_handle:
                    while chunk := source_handle.read(8 * 1024 * 1024):
                        digest.update(chunk)
                        output_handle.write(chunk)
                if args.keep_staged:
                    os.replace(temporary, destination)
                    temporary = destination
                    keep_temporary = True
            else:
                with temporary.open("rb") as staged_handle:
                    while chunk := staged_handle.read(8 * 1024 * 1024):
                        digest.update(chunk)
            if temporary.stat().st_size != size:
                raise OSError("staged archive byte size differs from source")
            result["sha256"] = digest.hexdigest()
            store = cellstream.open(temporary)
            try:
                actual_cells, actual_genes = map(int, store.shape)
                result["actual_cells"] = actual_cells
                result["actual_genes"] = actual_genes
                if actual_cells != expected:
                    raise ValueError(
                        f"cell count mismatch: expected {expected}, got {actual_cells}"
                    )
                if actual_genes != len(target_genes):
                    raise ValueError(
                        f"gene count mismatch: expected {len(target_genes)}, got {actual_genes}"
                    )
                if len(store.obs) != actual_cells or len(store.var) != actual_genes:
                    raise ValueError("obs/var dimensions do not match archive shape")
                archive_genes = store.var.index.astype(str).tolist()
                result["gene_order_matches"] = archive_genes == target_genes
                if archive_genes != target_genes:
                    mismatch = next(
                        (
                            index
                            for index, (actual, expected_gene) in enumerate(
                                zip(archive_genes, target_genes)
                            )
                            if actual != expected_gene
                        ),
                        min(len(archive_genes), len(target_genes)),
                    )
                    raise ValueError(
                        f"gene vocabulary/order mismatch at column {mismatch}"
                    )
                provenance = store.uns.get("celltok_stage1", {})
                archive_vocabulary_sha256 = provenance.get(
                    "gene_vocabulary_sha256", ""
                )
                result["archive_vocabulary_sha256"] = archive_vocabulary_sha256
                if archive_vocabulary_sha256 != vocabulary_sha256:
                    raise ValueError(
                        "archive gene-vocabulary SHA-256 differs from the formal vocabulary"
                    )
                if args.sample_rows and actual_cells:
                    indices = np.linspace(
                        0,
                        actual_cells - 1,
                        min(args.sample_rows, actual_cells),
                        dtype=np.int64,
                    ).tolist()
                    sample = store.gather_rows(indices, n_threads=1)
                    values = sample.data if hasattr(sample, "data") else np.asarray(sample)
                    if not np.all(np.isfinite(values)) or np.any(values < 0):
                        raise ValueError("sample contains non-finite or negative expression")
                result["status"] = "ok"
                total_cells += actual_cells
            finally:
                store.close()
        except Exception as exc:  # continue to produce a complete audit report
            failures += 1
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if temporary is not None:
                if result["status"] != "ok" or not keep_temporary:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
        report_rows.append(result)
        print(
            f"[{position}/{len(rows)}] {result['status']} {dataset_id}",
            flush=True,
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    summary = {
        "manifest": str(manifest),
        "archives": len(rows),
        "valid": len(rows) - failures,
        "failed": failures,
        "validated_cells": total_cells,
        "source_bytes": total_bytes,
        "gene_vocabulary": str(genes_path),
        "gene_vocabulary_sha256": vocabulary_sha256,
        "expected_genes": len(target_genes),
        "report": str(args.report.resolve()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
