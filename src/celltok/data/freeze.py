#!/usr/bin/env python3
"""Freeze complete primary datasets, replacing absent full files by complete shards."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from lib_vqvae.paths import resolve_data_path


def freeze_manifest(source, shard_manifest, output, require_all=False):
    source, shard_manifest, output = map(resolve_data_path, (source, shard_manifest, output))
    if output.exists() or output.with_suffix(".summary.json").exists():
        raise FileExistsError(f"refusing to replace frozen manifest: {output}")
    with source.open(newline="") as handle:
        original = list(csv.DictReader(handle))
    with shard_manifest.open(newline="") as handle:
        shards = list(csv.DictReader(handle))
    by_dataset = defaultdict(list)
    for row in shards:
        by_dataset[row["dataset_id"]].append(row)
    selected, excluded = [], []
    totals, target = Counter(), Counter()
    for row in original:
        if row["split"] not in {"train", "test"}:
            continue
        count = int(float(row["selected_cell_count"]))
        target[row["split"]] += count
        path = resolve_data_path(row["cellstream_path"], base_dir=source.parent)
        if path.is_file():
            pieces = [(row, path, count)]
        else:
            parts = sorted(by_dataset[row["dataset_id"]], key=lambda r: int(r["selected_cell_start"]))
            pieces = []
            cursor = 0
            for part in parts:
                start, stop = int(part["selected_cell_start"]), int(part["selected_cell_stop"])
                if (part["split"] != row["split"] or part["study_id"] != row["study_id"]
                        or part.get("cell_selection", "primary") != "primary"
                        or int(float(part["source_dataset_selected_cell_count"])) != count
                        or int(float(part["selected_cell_count"])) != stop-start):
                    raise ValueError(f"inconsistent shard provenance: {part['cellstream_path']}")
                part_path = resolve_data_path(part["cellstream_path"], base_dir=shard_manifest.parent)
                if start != cursor or stop <= start or not part_path.is_file():
                    break
                cursor = stop
                pieces.append((part, part_path, stop-start))
            if cursor != count or len(pieces) != len(parts):
                excluded.append({"dataset_id": row["dataset_id"], "split": row["split"], "cells": count})
                continue
        for part, part_path, part_cells in pieces:
            result = dict(row)
            result.update(part)
            result.update(cellstream_path=str(part_path), selected_cell_count=str(part_cells))
            selected.append(result)
            totals[row["split"]] += part_cells
    studies = defaultdict(set)
    for row in selected:
        studies[row["split"]].add(row["study_id"])
    overlap = studies["train"] & studies["test"]
    if overlap:
        raise ValueError(f"train/test study overlap: {sorted(overlap)}")
    if totals["train"] <= 0 or totals["test"] <= 0:
        raise ValueError("both train and test must be non-empty")
    if require_all and (excluded or totals != target):
        raise ValueError(f"full primary coverage required: {excluded}")
    paths = [r["cellstream_path"] for r in selected]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate archives in frozen manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in selected for key in row))
    with output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        "selection": "complete primary datasets only; incomplete datasets excluded in full",
        "source_manifest": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "manifest": str(output.resolve()),
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "selected_cells": dict(totals), "target_cells": dict(target),
        "all_target_cells_covered": not excluded and totals == target,
        "archives": len(selected),
        "datasets": {s: len({r['dataset_id'] for r in selected if r['split']==s}) for s in ('train', 'test')},
        "studies": {s: len(v) for s, v in studies.items()},
        "archive_bytes": sum(Path(r["cellstream_path"]).stat().st_size for r in selected),
        "excluded_incomplete_datasets": excluded,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/stage1/full_primary/manifest.csv"))
    parser.add_argument("--shard-manifest", type=Path, default=Path("data/stage1/full_primary/retry_large_shards.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true", help="Fail unless every train/test dataset is complete")
    arguments = parser.parse_args()
    freeze_manifest(arguments.manifest, arguments.shard_manifest, arguments.output, arguments.require_all)


if __name__ == "__main__":
    main()
