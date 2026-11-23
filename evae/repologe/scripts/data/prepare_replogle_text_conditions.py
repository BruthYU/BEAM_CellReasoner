"""Build versioned Replogle condition texts from obs metadata using CPU only.

No expression matrices, model weights, text encoders, or GPU libraries are read.
The generated cell index is a sidecar; existing training inputs are unchanged.
"""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np

from prepare_replogle_stage1 import ROOT, read_column, write_checked


VERSION = "replogle-text-v1"
CELL_LINES = {"K562": "K562", "hepg2": "HepG2", "jurkat": "Jurkat", "rpe1": "RPE1"}
CONTROL = "non-targeting"
PATTERN = (
    "Task: Single-cell Perturbation Prediction\n"
    "Cell Line: {cell_line}\n"
    "Perturbation Type: {perturbation_type}\n"
    "Target Gene: {target_gene}"
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()


def render(cell_line, dataset_label):
    control = dataset_label == CONTROL
    fields = {
        "cell_line": cell_line,
        "perturbation_type": "Non-targeting control" if control else "CRISPRi",
        "target_gene": "none" if control else dataset_label,
    }
    if any(not value or "\n" in value or "\r" in value for value in fields.values()):
        raise ValueError(f"Invalid condition fields: {fields}")
    text = PATTERN.format(**fields)
    text_sha = sha(text.encode())
    return fields, text, text_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/replogle_text_conditions/v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT / "artifacts/replogle_stage1/raw_audit.json"
    split_path = ROOT / "assets/scldm_replogle_test.json"
    audit = json.loads(audit_path.read_text())
    split = json.loads(split_path.read_text())
    assert audit["status"] == "passed"
    assert CONTROL not in split["pert"]
    test_cell_types, test_perts = set(split["cell_type"]), set(split["pert"])
    rows, files, indices = [], [], []
    totals = {"cells": 0, "train_cells": 0, "heldout_cells": 0, "control_cells": 0}
    seen_keys, seen_ids = set(), set()

    for file_index, metadata in enumerate(audit["files_in_loader_order"]):
        path = ROOT / metadata["path"]
        source_type = metadata["cell_type"]
        cell_line = CELL_LINES[source_type]
        mask_path = ROOT / metadata["kept_cell_mask"]
        mask_bytes = mask_path.read_bytes()
        keep = np.load(mask_path)
        assert sha(keep.tobytes()) == metadata["kept_cell_mask_sha256"]
        assert keep.dtype == np.bool_ and keep.shape == (metadata["raw_cells"],)
        source_stat = path.stat()
        assert source_stat.st_size == metadata["raw_size_bytes"]
        assert source_stat.st_mtime_ns == metadata["raw_mtime_ns"], "Re-audit changed source files"
        # Read labels only. In particular, never materialise the H5AD X matrix.
        with h5py.File(path, "r") as source:
            obs = source["obs"]
            labels = np.asarray(read_column(obs, "gene"), dtype=str)
            barcode_name = obs.attrs.get("_index", "_index")
            barcodes = np.asarray(read_column(obs, barcode_name), dtype=str)
            assert len(labels) == len(barcodes) == len(keep)
            assert len(np.unique(barcodes)) == len(barcodes)
            if "cell_type" in obs:
                observed_types = np.asarray(read_column(obs, "cell_type"), dtype=str)
                assert set(observed_types.tolist()) == {source_type}
        kept_rows = np.flatnonzero(keep).astype(np.int32)
        kept_labels = labels[keep]
        unique, inverse, counts = np.unique(kept_labels, return_inverse=True, return_counts=True)
        condition_indices = np.empty(len(unique), dtype=np.int32)
        heldout_by_label = np.zeros(len(unique), dtype=np.bool_)
        file_control = 0
        for label_index, (label, count) in enumerate(zip(unique.tolist(), counts.tolist())):
            key = (cell_line, label)
            assert key not in seen_keys, "Expected one input file per cell line"
            seen_keys.add(key)
            fields, text, text_sha = render(cell_line, label)
            condition_id = VERSION + "-" + text_sha[:16]
            assert condition_id not in seen_ids, "Condition ID collision"
            seen_ids.add(condition_id)
            is_heldout = source_type in test_cell_types and label in test_perts
            is_control = label == CONTROL
            assert not (is_control and is_heldout)
            row = {
                "condition_row": len(rows), "condition_id": condition_id,
                "template_version": VERSION, "source_file_index": file_index,
                "source_cell_type": source_type, "dataset_label": label,
                **fields, "is_control": is_control,
                "split": "heldout" if is_heldout else "train", "n_cells": count,
                "n_train_cells": 0 if is_heldout else count,
                "n_heldout_cells": count if is_heldout else 0,
                "text_sha256": text_sha, "text": text,
            }
            condition_indices[label_index] = row["condition_row"]
            heldout_by_label[label_index] = is_heldout
            rows.append(row)
            totals["cells"] += count
            totals["train_cells"] += row["n_train_cells"]
            totals["heldout_cells"] += row["n_heldout_cells"]
            totals["control_cells"] += count if is_control else 0
            file_control += count if is_control else 0
        mapped = condition_indices[inverse]
        heldout = heldout_by_label[inverse]
        assert len(mapped) == metadata["kept_cells"]
        assert int(heldout.sum()) == metadata["heldout_cells"]
        assert file_control == metadata["control_cells_after_qc"]
        indices.append({"source_file_index": np.full(len(mapped), file_index, dtype=np.uint8),
                        "source_row": kept_rows, "condition_row": mapped, "is_heldout": heldout})
        files.append({
            "source_file_index": file_index, "path": metadata["path"],
            "resolved_path": str(path.resolve()), "source_cell_type": source_type,
            "cell_line": cell_line, "raw_size_bytes": source_stat.st_size,
            "raw_mtime_ns": source_stat.st_mtime_ns,
            "obs_labels_sha256": sha(("\n".join(labels.tolist()) + "\n").encode()),
            "obs_barcodes_sha256": sha(("\n".join(barcodes.tolist()) + "\n").encode()),
            "kept_cell_mask": metadata["kept_cell_mask"],
            "kept_cell_mask_sha256": sha(keep.tobytes()),
            "kept_cell_mask_file_sha256": sha(mask_bytes), "cells": len(mapped),
            "conditions": len(unique), "control_cells": file_control,
            "train_cells": int((~heldout).sum()), "heldout_cells": int(heldout.sum()),
        })
        print(json.dumps(files[-1], ensure_ascii=False), flush=True)

    assert totals["cells"] == audit["total_cells_after_qc"]
    assert totals["train_cells"] == audit["train_cells"]
    assert totals["heldout_cells"] == audit["heldout_cells"]
    assert sum(row["is_control"] for row in rows) == len(files)
    assert sum(row["split"] == "heldout" for row in rows) == len(test_perts)
    assert len({row["text"] for row in rows}) == len(rows)
    merged = {name: np.concatenate([part[name] for part in indices]) for name in indices[0]}
    # Validate full cell-to-text coverage and group counts, not just examples.
    grouped = np.bincount(merged["condition_row"], minlength=len(rows))
    np.testing.assert_array_equal(grouped, [row["n_cells"] for row in rows])
    np.testing.assert_array_equal(
        np.asarray([row["split"] == "heldout" for row in rows])[merged["condition_row"]],
        merged["is_heldout"],
    )
    by_key = {(row["cell_line"], row["dataset_label"]): row for row in rows}
    examples = [by_key[("HepG2", label)] for label in ["EIF2S1", CONTROL]]
    # Preserve the already demonstrated prompt byte-for-byte.
    prior_case = ROOT / "artifacts/replogle_text_conditions/case_hepg2_non_targeting_row16/case.json"
    if prior_case.exists():
        assert examples[1]["text"] == json.loads(prior_case.read_text())["prompt"].rstrip("\n")

    outputs = {
        "template.json": json_bytes({
            "version": VERSION, "pattern": PATTERN,
            "cell_line_names": CELL_LINES, "control_dataset_label": CONTROL,
            "target_gene_policy": "Use the existing dataset label verbatim; controls render as none",
            "text_format": "UTF-8, LF separators, no trailing newline inside text",
            "generation_count": "Sampling parameter outside condition text",
        }),
        "conditions.jsonl": ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode(),
        "examples.json": json_bytes(examples),
    }
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    outputs["conditions.csv"] = csv_buffer.getvalue().encode()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **merged)
    outputs["cell_condition_index.npz"] = buffer.getvalue()
    for filename, data in outputs.items():
        write_checked(output / filename, data)

    # Round-trip both tabular formats and the saved index before accepting them.
    saved = [json.loads(line) for line in (output / "conditions.jsonl").read_text().splitlines()]
    assert saved == rows
    with (output / "conditions.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [(row["condition_id"], row["text"]) for row in csv_rows] == [
        (row["condition_id"], row["text"]) for row in rows]
    with np.load(output / "cell_condition_index.npz") as saved_index:
        for name, value in merged.items():
            np.testing.assert_array_equal(saved_index[name], value)

    report = {
        "status": "passed", "template_version": VERSION,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "raw_audit_sha256": sha(audit_path.read_bytes()),
        "split_sha256": sha(split_path.read_bytes()),
        "device": "cpu", "read_expression_matrix": False, "embeddings_computed": False,
        "training_integration": "Sidecar only; existing AR inputs and jobs are unchanged",
        "condition_count": len(rows),
        "train_condition_count": sum(row["split"] == "train" for row in rows),
        "heldout_condition_count": sum(row["split"] == "heldout" for row in rows),
        "unique_dataset_labels": len({row["dataset_label"] for row in rows}),
        **totals, "files": files,
        "index_contract": {
            "order": "Files in raw audit loader order; retained source rows ascending within each file",
            "source_row": "Zero-based physical H5AD row, not a cell barcode",
            "condition_row": "Zero-based row in conditions.jsonl (CSV data row excluding header)",
            "source_file_index": "Index into this report's files array",
            "is_heldout": "Existing scLDM cell-line x perturbation holdout; not used in text",
        },
        "validation": "All retained cells mapped exactly once; counts/splits match raw audit; CSV/JSONL/NPZ round trips passed",
        "outputs": {filename: {"sha256": sha(data), "bytes": len(data)}
                    for filename, data in outputs.items()},
    }
    (output / "manifest.json").write_bytes(json_bytes(report))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"files", "index_contract", "outputs"}}, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
