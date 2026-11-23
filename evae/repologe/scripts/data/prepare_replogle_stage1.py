"""Prepare raw-file aliases and the published scLDM gene order for Stage 1."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "0e03e404b096636a59493803d0bdc28ead9a3520"
SOURCE_URL = (
    "https://raw.githubusercontent.com/czi-ai/scLDM/"
    f"{SOURCE_COMMIT}/metadata/replogle_test.json"
)
RAW_DIR = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/"
    "datasets/Replogle_Nadig"
)
FILES = (
    ("K562_essential_raw_singlecell_01.h5ad", "K562_essential_raw_singlecell_01.h5ad", "K562"),
    ("GSE264667_hepg2_raw_singlecell_01.h5ad", "hepg2_raw_singlecell_01.h5ad", "hepg2"),
    ("GSE264667_jurkat_raw_singlecell_01.h5ad", "jurkat_raw_singlecell_01.h5ad", "jurkat"),
    ("rpe1_raw_singlecell_01.h5ad", "rpe1_raw_singlecell_01.h5ad", "rpe1"),
)


def read_column(frame, name):
    """Read categorical columns in both legacy and current H5AD encodings."""
    node = frame[name]
    if isinstance(node, h5py.Group):
        categories, codes = node["categories"].asstr()[:], node["codes"][:]
    elif "__categories" in frame and name in frame["__categories"]:
        categories, codes = frame["__categories"][name].asstr()[:], node[:]
    else:
        return node.asstr()[:] if node.dtype.kind in "OSU" else node[:]
    if np.any(codes < 0):
        raise ValueError(f"Missing values in required column {name!r}")
    return categories[codes]


def write_checked(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace a different existing asset: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    panel_path = ROOT / "assets/replogle_scldm2k_source.json"
    if panel_path.exists():
        source_bytes = panel_path.read_bytes()
    else:
        request = Request(SOURCE_URL, headers={"User-Agent": "sc-evae-reproduction"})
        with urlopen(request, timeout=30) as response:
            source_bytes = response.read()
    panel = json.loads(source_bytes)
    genes = panel["genes"]
    split = json.loads((ROOT / "assets/scldm_replogle_test.json").read_text())
    assert len(genes) == len(set(genes)) == 2000
    assert set(panel["labels"]["gene"]) == set(split["pert"])
    assert panel["labels"]["cell_line"] == split["cell_type"] == ["hepg2"]

    symbol_to_ids = {}
    file_reports = []
    for source_name, alias, cell_type in FILES:
        source = args.raw_dir / source_name
        with h5py.File(source, "r") as data:
            names = read_column(data["var"], "gene_name")
            ensembl = data["var"][data["var"].attrs.get("_index", "_index")].asstr()[:]
            perts = read_column(data["obs"], "gene")
            for symbol, gene_id in zip(names, ensembl):
                symbol_to_ids.setdefault(str(symbol), set()).add(str(gene_id))
            missing = sorted(set(genes) - set(names))
            if missing:
                raise ValueError(f"{source_name}: canonical genes missing: {missing}")
            report = {
                "source": str(source), "alias": alias, "cell_type": cell_type,
                "n_cells": int(data["X"].shape[0]),
                "n_native_genes": int(data["X"].shape[1]),
                "canonical_gene_matches": 2000,
                "raw_holdout_cells": int(np.isin(perts, split["pert"]).sum())
                if cell_type == "hepg2" else 0,
                "control_cells": int((perts == "non-targeting").sum()),
                "size_bytes": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns,
            }
            file_reports.append(report)
    ambiguous = {g: sorted(symbol_to_ids.get(g, set())) for g in genes
                 if len(symbol_to_ids.get(g, set())) != 1}
    if ambiguous:
        raise ValueError(f"Missing or ambiguous gene identifiers: {ambiguous}")
    ordered_ids = [next(iter(symbol_to_ids[g])) for g in genes]
    assert len(set(ordered_ids)) == 2000

    alias_dir = ROOT / "datasets/replogle"
    alias_dir.mkdir(parents=True, exist_ok=True)
    for report in file_reports:
        alias_path = alias_dir / report["alias"]
        source = Path(report["source"]).resolve()
        if alias_path.is_symlink() or alias_path.exists():
            if not alias_path.is_symlink() or alias_path.resolve() != source:
                raise RuntimeError(f"Refusing to replace existing path: {alias_path}")
        else:
            alias_path.symlink_to(source)
    write_checked(panel_path, source_bytes)
    symbols_bytes = ("\n".join(genes) + "\n").encode()
    ids_bytes = ("\n".join(ordered_ids) + "\n").encode()
    write_checked(ROOT / "assets/replogle_scldm2k_gene_order.symbols.txt", symbols_bytes)
    write_checked(ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt", ids_bytes)
    provenance = {
        "source_url": SOURCE_URL, "source_commit": SOURCE_COMMIT,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "symbols_sha256": hashlib.sha256(symbols_bytes).hexdigest(),
        "ensembl_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        "gene_mapping": dict(zip(genes, ordered_ids)),
        "files_in_loader_order": file_reports,
        "raw_total_cells": sum(f["n_cells"] for f in file_reports),
        "raw_holdout_cells": sum(f["raw_holdout_cells"] for f in file_reports),
        "raw_files_opened_read_only": True,
    }
    destination = ROOT / "artifacts/replogle_stage1/data_preparation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if k != "gene_mapping"}, indent=2))


if __name__ == "__main__":
    main()
