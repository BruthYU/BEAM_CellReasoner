"""Align historical Replogle labels to existing ESM vectors via HGNC identifiers.

Adds no learned or synthesized protein features. Ambiguous matches are skipped;
the original loader's zero-vector fallback handles unresolved labels.
"""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import csv
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    artifacts = ROOT / "artifacts/replogle_stage2"
    raw_report_path = artifacts / "raw_embedding_audit.json"
    # Preserve the first preparation receipt made before alignment was added.
    if not raw_report_path.exists():
        original = json.loads((artifacts / "embedding_audit.json").read_text())
        assert "symbol_alignment" not in original
        raw_report_path.write_text(json.dumps(original, indent=2) + "\n")
    raw_report = json.loads(raw_report_path.read_text())
    raw = torch.load(raw_report["path"], map_location="cpu", weights_only=True)
    annotation = artifacts / "hgnc_complete_set.txt"
    with annotation.open() as handle:
        records = [row for row in csv.DictReader(handle, delimiter="\t") if row["status"] == "Approved"]
    current, previous, aliases = {}, {}, {}
    for i, row in enumerate(records):
        current.setdefault(row["symbol"], set()).add(i)
        for field, lookup in [("prev_symbol", previous), ("alias_symbol", aliases)]:
            for name in row[field].split("|"):
                if name:
                    lookup.setdefault(name, set()).add(i)
    aligned, mappings, unresolved = {}, [], []
    for name in raw_report["required_labels"]:
        if name in raw:
            aligned[name] = raw[name]
            continue
        matching_records = current.get(name) or previous.get(name) or aliases.get(name) or set()
        if len(matching_records) != 1:
            unresolved.append(name)
            continue
        row = records[next(iter(matching_records))]
        if row["symbol"] in raw:
            source = row["symbol"]
        else:
            candidates = set((row["prev_symbol"] + "|" + row["alias_symbol"]).split("|")) & set(raw)
            if len(candidates) != 1:
                unresolved.append(name)
                continue
            source = next(iter(candidates))
        aligned[name] = raw[source]
        mappings.append({"dataset_label": name, "embedding_key": source,
                         "hgnc_id": row["hgnc_id"], "approved_symbol": row["symbol"],
                         "ensembl_gene_id": row["ensembl_gene_id"]})
    heldout = set(json.loads((ROOT / "assets/scldm_replogle_test.json").read_text())["pert"])
    zeros = sorted(name for name, value in aligned.items() if not value.count_nonzero())
    missing_heldout = sorted(heldout & (set(unresolved) | set(zeros)))
    assert not missing_heldout, f"Held-out targets still lack protein features: {missing_heldout}"
    for name, value in aligned.items():
        assert torch.isfinite(value).all()
    destination = ROOT / "datasets/competition_support_set/ESM2_replogle_symbol_aligned.pt"
    temporary = destination.with_suffix(".pt.part")
    torch.save(aligned, temporary)
    temporary.replace(destination)
    report = dict(raw_report)
    report.update(path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        embedding_keys=len(aligned), matched_labels=len(aligned), missing_labels=unresolved,
        present_but_zero_labels=zeros, heldout_missing_or_zero=missing_heldout,
        symbol_alignment={"source_url": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
            "hgnc_snapshot_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
            "policy": "Exact existing keys first, then unambiguous HGNC approved/previous/alias identity; copy existing vectors unchanged",
            "mapped_labels": mappings},
        paper_feature_spec_discrepancy="Paper states ESM2-3B (2560 dims); original repository config points at Arc features with 5120 dims",
        status="passed")
    (artifacts / "embedding_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": str(destination), "feature_dim": report["feature_dim"],
        "matched_labels": len(aligned), "hgnc_mappings": len(mappings), "missing_labels": unresolved,
        "heldout_missing_or_zero": missing_heldout, "mappings": mappings}, indent=2), flush=True)


if __name__ == "__main__":
    main()
