#!/usr/bin/env python3
"""Convert a source H5AD into a 5,000-gene CellStream archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from cellstream.cell import CellArchiveWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Source H5AD")
    parser.add_argument("--output", required=True, type=Path, help="Output .csad")
    parser.add_argument(
        "--genes", type=Path, default=Path("data/5000_gene_names.csv")
    )
    parser.add_argument(
        "--matrix",
        default="raw",
        help=(
            "raw, raw-or-X, X, or layer:<name>; raw-or-X is recommended for "
            "mixed CELLXGENE source H5ADs and still validates integer counts"
        ),
    )
    parser.add_argument(
        "--storage",
        choices=["auto", "counts", "float32"],
        default="auto",
        help=(
            "CellStream value representation. auto uses integer counts for raw and "
            "float32 for X/layers; use float32 to reproduce the released blood tutorial."
        ),
    )
    parser.add_argument(
        "--gene-column",
        default="feature_name",
        help="Source var column containing gene symbols; falls back to var_names",
    )
    parser.add_argument("--block-rows", type=int, default=8192)
    parser.add_argument("--encode-threads", type=int, default=None)
    parser.add_argument(
        "--selected-cell-start",
        type=int,
        default=0,
        help=(
            "Zero-based start within the cells retained by --cell-selection. "
            "Use with --selected-cell-stop to create resumable CellStream shards."
        ),
    )
    parser.add_argument(
        "--selected-cell-stop",
        type=int,
        default=None,
        help="Exclusive stop within the cells retained by --cell-selection.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cell-selection",
        choices=["primary", "non-primary", "all"],
        default="primary",
        help=(
            "Select rows using obs.is_primary_data. 'non-primary' writes only the "
            "complement needed for an all-rows delta dataset; default: primary"
        ),
    )
    selection.add_argument(
        "--all-cells",
        action="store_true",
        help="Deprecated alias for --cell-selection all",
    )
    parser.add_argument("--group-by", default=None)
    parser.add_argument("--payload-checksum", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_target_genes(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "gene" not in frame:
        raise ValueError(f"{path} must contain a 'gene' column")
    genes = frame["gene"].astype(str).tolist()
    if len(genes) != len(set(genes)):
        raise ValueError(f"target gene vocabulary contains duplicates: {path}")
    return genes


def sanitize_obs_columns(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Make observation column names safe and unique as HDF5 group keys.

    CELLXGENE source metadata occasionally contains names such as
    ``Region/Tooth #``. AnnData interprets a slash as an HDF5 path separator,
    so CellStream cannot embed that metadata verbatim. The mapping is recorded
    in archive provenance to make the rename reversible.
    """

    safe_names: list[str] = []
    used: set[str] = set()
    renames: list[dict[str, object]] = []
    for position, value in enumerate(frame.columns):
        original = str(value)
        base = original.replace("/", "__slash__").replace("\x00", "__nul__")
        if not base:
            base = "unnamed"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}__{suffix}"
            suffix += 1
        used.add(candidate)
        safe_names.append(candidate)
        if candidate != original:
            renames.append(
                {
                    "position": position,
                    "original": original,
                    "sanitized": candidate,
                }
            )
    result = frame.copy()
    result.columns = safe_names
    return result, renames


def select_matrix(adata, matrix_name: str):
    if matrix_name in {"raw", "raw-or-X"}:
        if adata.raw is None:
            if matrix_name == "raw":
                raise ValueError("--matrix raw requested, but the H5AD has no .raw")
            return adata.X, adata.var, "X"
        return adata.raw.X, adata.raw.var, "raw"
    if matrix_name == "X":
        return adata.X, adata.var, "X"
    if matrix_name.startswith("layer:"):
        layer = matrix_name.split(":", 1)[1]
        if layer not in adata.layers:
            raise ValueError(f"layer {layer!r} not found in the H5AD")
        return adata.layers[layer], adata.var, matrix_name
    raise ValueError("--matrix must be raw, raw-or-X, X, or layer:<name>")


def build_gene_mapping(
    source_var: pd.DataFrame, target_genes: list[str], gene_column: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if gene_column in source_var:
        source_names = source_var[gene_column].astype(str).to_numpy()
    else:
        source_names = source_var.index.astype(str).to_numpy()

    target_positions = {gene: index for index, gene in enumerate(target_genes)}
    source_columns: list[int] = []
    output_columns: list[int] = []
    present: set[str] = set()
    for source_index, gene in enumerate(source_names):
        output_index = target_positions.get(gene)
        if output_index is not None:
            source_columns.append(source_index)
            output_columns.append(output_index)
            present.add(gene)
    missing = [gene for gene in target_genes if gene not in present]
    return (
        np.asarray(source_columns, dtype=np.int64),
        np.asarray(output_columns, dtype=np.int64),
        missing,
    )


def align_expression_block(
    matrix,
    row_slice: slice,
    source_columns: np.ndarray,
    output_columns: np.ndarray,
    n_genes: int,
    keep_rows: np.ndarray,
    value_dtype: np.dtype,
) -> sp.csr_matrix:
    # AnnData's backed CSR path only supports a slice on the minor axis.
    # Combining a row slice with a column array calls to_memory(), loading the
    # entire source matrix for every block. Materialize just the contiguous
    # row block before selecting columns in memory.
    if getattr(matrix, "format", None) == "csr":
        selected = matrix[row_slice, :][:, source_columns]
    else:
        selected = matrix[row_slice, source_columns]
    selected = sp.coo_matrix(selected)
    if keep_rows.size != selected.shape[0]:
        raise ValueError("internal row-filter length mismatch")
    if not np.all(keep_rows):
        row_remap = np.full(keep_rows.size, -1, dtype=np.int64)
        row_remap[keep_rows] = np.arange(int(keep_rows.sum()))
        entry_keep = keep_rows[selected.row]
        rows = row_remap[selected.row[entry_keep]]
        columns = output_columns[selected.col[entry_keep]]
        data = selected.data[entry_keep]
        n_rows = int(keep_rows.sum())
    else:
        rows = selected.row
        columns = output_columns[selected.col]
        data = selected.data
        n_rows = selected.shape[0]

    if data.size and np.issubdtype(value_dtype, np.integer):
        if not (
            np.all(np.isfinite(data))
            and np.all(data >= 0)
            and np.all(data == np.rint(data))
            and float(data.max()) <= np.iinfo(np.uint32).max
        ):
            raise ValueError(
                "selected matrix is not non-negative integer counts; use "
                "--storage float32 or select a raw/counts matrix"
            )
    aligned = sp.coo_matrix(
        (data.astype(value_dtype, copy=False), (rows, columns)),
        shape=(n_rows, n_genes),
        dtype=value_dtype,
    ).tocsr()
    aligned.sum_duplicates()
    aligned.sort_indices()
    return aligned


def main() -> None:
    args = parse_args()
    if args.block_rows < 1:
        raise ValueError("--block-rows must be >= 1")
    if args.selected_cell_start < 0:
        raise ValueError("--selected-cell-start must be >= 0")
    if (
        args.selected_cell_stop is not None
        and args.selected_cell_stop <= args.selected_cell_start
    ):
        raise ValueError("--selected-cell-stop must be greater than the start")
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    genes_path = args.genes.resolve()
    target_genes = read_target_genes(genes_path)

    started = time.monotonic()

    def progress(stage: str, **fields) -> None:
        record = {
            "stage": stage,
            "output": str(output_path),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            **fields,
        }
        print(json.dumps(record), file=sys.stderr, flush=True)

    progress("opening_source")
    adata = ad.read_h5ad(input_path, backed="r")
    try:
        matrix, source_var, selected_matrix = select_matrix(adata, args.matrix)
        storage = args.storage
        if storage == "auto":
            storage = (
                "counts" if args.matrix in {"raw", "raw-or-X"} else "float32"
            )
        value_dtype = np.dtype(np.uint32 if storage == "counts" else np.float32)
        codec = "pfordelta" if storage == "counts" else "zstd"
        source_columns, output_columns, missing = build_gene_mapping(
            source_var, target_genes, args.gene_column
        )
        if source_columns.size == 0:
            raise ValueError("none of the target genes occur in the source H5AD")

        cell_selection = "all" if args.all_cells else args.cell_selection
        if cell_selection == "all":
            selected_rows = np.ones(adata.n_obs, dtype=bool)
        elif "is_primary_data" not in adata.obs:
            if cell_selection == "non-primary":
                raise ValueError(
                    "--cell-selection non-primary requires obs.is_primary_data"
                )
            # Preserve the historical behavior for sources without this optional
            # CELLxGENE annotation: all rows are treated as primary.
            selected_rows = np.ones(adata.n_obs, dtype=bool)
        else:
            is_primary = adata.obs["is_primary_data"].astype(bool).to_numpy()
            selected_rows = is_primary if cell_selection == "primary" else ~is_primary
        selected_indices = np.flatnonzero(selected_rows)
        if selected_indices.size == 0:
            raise ValueError(
                f"no cells remain after cell selection {cell_selection!r}; exclude "
                "this source H5AD from the selected manifest"
            )
        selected_total = int(selected_indices.size)
        selected_stop = (
            selected_total
            if args.selected_cell_stop is None
            else args.selected_cell_stop
        )
        if args.selected_cell_start >= selected_total or selected_stop > selected_total:
            raise ValueError(
                "selected-cell range "
                f"[{args.selected_cell_start}, {selected_stop}) is outside "
                f"the {selected_total} cells retained by {cell_selection!r}"
            )
        shard_indices = selected_indices[args.selected_cell_start:selected_stop]
        progress(
            "source_ready",
            matrix_format=getattr(matrix, "format", "dense"),
            source_cells=adata.n_obs,
            shard_cells=int(shard_indices.size),
            selected_cell_start=args.selected_cell_start,
            selected_cell_stop=selected_stop,
        )
        selected_rows = np.zeros(adata.n_obs, dtype=bool)
        selected_rows[shard_indices] = True
        output_obs = adata.obs.iloc[shard_indices].copy()
        output_obs, obs_column_renames = sanitize_obs_columns(output_obs)
        output_var = pd.DataFrame(index=pd.Index(target_genes, name="gene"))
        output_var["feature_name"] = target_genes
        vocabulary_sha256 = hashlib.sha256(genes_path.read_bytes()).hexdigest()
        provenance = {
            "celltok_stage1": {
                "source_h5ad": str(input_path),
                "source_matrix": selected_matrix,
                "gene_vocabulary": str(genes_path),
                "gene_vocabulary_sha256": vocabulary_sha256,
                "target_gene_count": len(target_genes),
                "source_genes_mapped": int(source_columns.size),
                "target_genes_missing": len(missing),
                "obs_column_renames_json": json.dumps(
                    obs_column_renames, ensure_ascii=False, separators=(",", ":")
                ),
                "storage": storage,
                "stores_raw_counts": storage == "counts",
                "normalization_deferred_to_dataloader": True,
                "cell_selection": cell_selection,
                "selected_cell_start": args.selected_cell_start,
                "selected_cell_stop": selected_stop,
                "selected_cell_total": selected_total,
            }
        }

        with CellArchiveWriter(
            output_path,
            n_vars=len(target_genes),
            value_dtype=value_dtype,
            codec=codec,
            threads=args.encode_threads,
            overwrite=args.overwrite,
            payload_checksum=args.payload_checksum,
        ) as writer:
            scan_start = int(shard_indices[0])
            scan_stop = int(shard_indices[-1]) + 1
            written_cells = 0
            last_report = time.monotonic()
            progress("writing_blocks", source_row_start=scan_start, source_row_stop=scan_stop)
            for start in range(scan_start, scan_stop, args.block_rows):
                stop = min(start + args.block_rows, scan_stop)
                if not np.any(selected_rows[start:stop]):
                    continue
                block = align_expression_block(
                    matrix,
                    slice(start, stop),
                    source_columns,
                    output_columns,
                    len(target_genes),
                    selected_rows[start:stop],
                    value_dtype,
                )
                if block.shape[0]:
                    writer.add_block(block)
                    written_cells += block.shape[0]
                if time.monotonic() - last_report >= 30:
                    progress(
                        "writing_blocks",
                        source_row=stop,
                        source_row_stop=scan_stop,
                        written_cells=written_cells,
                        shard_cells=int(shard_indices.size),
                    )
                    last_report = time.monotonic()
            if written_cells != len(output_obs):
                raise RuntimeError(f"written cell mismatch: {written_cells} != {len(output_obs)}")
            progress("finalizing", written_cells=written_cells)
            writer.finalize(
                obs=output_obs,
                var=output_var,
                uns=provenance,
                group_by=args.group_by,
            )
            progress("complete", written_cells=written_cells)
    finally:
        adata.file.close()

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "cells": int(len(output_obs)),
        "genes": len(target_genes),
        "source_gene_columns": int(source_columns.size),
        "missing_target_genes": len(missing),
        "source_matrix": selected_matrix,
        "storage": storage,
        "cell_selection": cell_selection,
        "selected_cell_start": args.selected_cell_start,
        "selected_cell_stop": selected_stop,
        "selected_cell_total": selected_total,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
