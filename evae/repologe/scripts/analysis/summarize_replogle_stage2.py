"""Collect the nine paper metrics without mixing token loss with generation scores."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/replogle_stage2_comparison"
RUNS = {
    "mse": ROOT / "outputs/replogle_stage2/replogle-stage2-ar-fsq-mse-seed420",
    "nb": ROOT / "outputs/replogle_experiments/ar-fsq-nb-l64-k512-990fc9c87a-2gpu-0868/stage2",
}
METRICS = [
    ("W2", "w2_pca30", "lower"),
    ("MMD2", "mmd2_rbf_scldm", "lower"),
    ("FD", "fd_pca30", "lower"),
    ("PDS", "discrimination_score_l1", "higher"),
    ("PR-AUC", "pr_auc", "higher"),
    ("Sp-sig", "de_spearman_sig", "higher"),
    ("Sp-LFC", "de_spearman_lfc_sig", "higher"),
    ("ovl@N", "overlap_at_n", "higher"),
    ("P-Delta", "pearson_delta", "higher"),
]


def main():
    reference = json.loads((ART / "paper_table1_reference.json").read_text())
    summaries, audits = {}, {}
    for head, run in RUNS.items():
        summaries[head] = json.loads((run / "eval_results/summary_metrics.json").read_text())
        audits[head] = json.loads((run / "eval_results/pds_target_gene_audit.json").read_text())
    rows = []
    for label, key, direction in METRICS:
        row = {"metric": label, "key": key, "better": direction,
               "n_unit": "cells_per_population" if direction == "lower" else "perturbations"}
        for head, paper_key in [("mse", "ar_fsq_mse"), ("nb", "ar_fsq_nb_sample")]:
            paper = reference[paper_key][key]
            if direction == "lower":
                paper = reference["table6_distribution"][paper_key][key]["mean"]
                value = summaries[head]["distribution"][key]
                row[f"paper_{head}_se"] = reference["table6_distribution"][paper_key][key]["se"]
                row[f"ours_{head}_n"] = 5000
            elif key == "discrimination_score_l1":
                value = audits[head]["symbol_aligned_target_excluded"]["mean"]
                row[f"ours_{head}_unadjusted_pds"] = summaries[head]["cell_eval"][key]["mean"]
                row[f"ours_{head}_n"] = audits[head]["symbol_aligned_target_excluded"]["n"]
            elif key == "de_spearman_sig":
                value = summaries[head]["cell_eval"][key]["value"]
                row[f"ours_{head}_n"] = summaries[head]["cell_eval"][key]["n_perts"]
            else:
                value = summaries[head]["cell_eval"][key]["mean"]
                row[f"ours_{head}_n"] = summaries[head]["cell_eval"][key]["n"]
            row[f"paper_{head}"] = paper
            row[f"ours_{head}"] = value
            row[f"ours_{head}_minus_paper"] = value - paper
        rows.append(row)
    metadata = {
        "models": {head: str(run.relative_to(ROOT)) for head, run in RUNS.items()},
        "checkpoint": "step_0075000_final", "evaluation_seed": 420,
        "paper_sources": [reference["source_url"], reference["table6_distribution"]["source_url"]],
        "population": "28177 HepG2 cells, 372 held-out target perturbations",
        "mmd_definition": "Biased multiscale RBF mixture MMD squared, PCA30 fit on true cells",
        "normalization": "log1p CP10K within the fixed official 2000-gene panel",
        "pds": "Original PDS function, symbol-aligned target-gene exclusion; original raw scores retained",
        "aggregation": "One evaluation seed per trained model; paper reports four-seed mean and SE",
        "metrics": rows,
        "source_summary_sha256": {head: hashlib.sha256((run / "eval_results/summary_metrics.json").read_bytes()).hexdigest()
                                  for head, run in RUNS.items()},
    }
    (ART / "paper_comparison.json").write_text(json.dumps(metadata, indent=2) + "\n")
    columns = ["metric", "better", "paper_mse", "ours_mse", "paper_nb", "ours_nb", "ours_mse_n", "ours_nb_n", "n_unit"]
    with (ART / "paper_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
