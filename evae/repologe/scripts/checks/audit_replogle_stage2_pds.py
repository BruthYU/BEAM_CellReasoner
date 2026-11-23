"""Audit target-gene exclusion using the original PDS function and saved DE tables."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sc_evae.metrics.state_metrics import aggregate_per_pert, discrimination_score_l1

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    output = ROOT / args.run_dir / "eval_results"
    genes = (ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt").read_text().splitlines()
    symbols = (ROOT / "assets/replogle_scldm2k_gene_order.symbols.txt").read_text().splitlines()
    assert len(genes) == len(symbols) == 2000
    real = pd.read_csv(output / "real_de_table.csv.gz")
    pred = pd.read_csv(output / "pred_de_table.csv.gz")
    labels = sorted(real.target.unique())
    assert set(labels) == set(pred.target.unique())
    # DE tables store expm1(mean(log1p expression)); invert to exactly the
    # log-space pseudobulks consumed by the original PDS implementation.
    def pseudobulk(table):
        matrix = table.pivot(index="target", columns="gene", values="mean_pert")
        return np.log1p(matrix.loc[labels, genes].to_numpy())
    ctrl = real.loc[real.target == labels[0]].set_index("gene").loc[genes, "mean_ctrl"]
    ctrl_pb = np.log1p(ctrl.to_numpy())
    real_pb, pred_pb = pseudobulk(real), pseudobulk(pred)
    kwargs = dict(metric="l1", exclude_target_gene=True)
    original = discrimination_score_l1(real_pb, pred_pb, ctrl_pb, labels, gene_names=genes, **kwargs)
    corrected = discrimination_score_l1(real_pb, pred_pb, ctrl_pb, labels, gene_names=symbols, **kwargs)
    saved = json.loads((output / "state_per_pert.json").read_text())
    max_error = max(abs(original[label] - saved[label]["discrimination_score_l1"]) for label in labels)
    assert max_error < 1e-10, "Recovered pseudobulk PDS differs from the original evaluation"
    report = {
        "original": aggregate_per_pert(original),
        "symbol_aligned_target_excluded": aggregate_per_pert(corrected),
        "original_pds_roundtrip_max_error": max_error,
        "heldout_target_genes_present_in_panel": len(set(labels) & set(symbols)),
        "heldout_target_genes_matching_ensembl_ids": len(set(labels) & set(genes)),
        "method": "Original PDS function; map gene identifiers to official panel symbols for target exclusion",
        "source_files_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                for name in ["real_de_table.csv.gz", "pred_de_table.csv.gz"]},
        "per_pert_symbol_aligned_pds": corrected,
    }
    path = output / "pds_target_gene_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    summary = json.loads((output / "summary_metrics.json").read_text())
    summary["cell_eval"]["discrimination_score_l1"] = report["symbol_aligned_target_excluded"]
    (output / "summary_metrics_symbol_aligned.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if not key.startswith("per_pert")}, indent=2))


if __name__ == "__main__":
    main()
