"""Frozen evaluation rows and metric adapters for independent A/B experiments."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import json
from pathlib import Path

import numpy as np
import torch

from replogle_llm_common import ROOT, read_json, save_json, sha
from replogle_control_common import ART, CellStore, check_inputs

EVAL_ART = ART / "evaluation_observed_depth_v1"
EVAL_ROOT = ROOT / "outputs/replogle_llm_control_evaluation" / "ab-observed-depth-v1"
SCRIPTS = ['replogle_control_eval_common.py', 'check_replogle_control_eval.py', 'run_replogle_control_eval_worker.sh', 'submit_replogle_control_eval.py', 'eval_replogle_qwen.py', 'replogle_qwen_legacy_eval.py', 'replogle_qwen_inference.py', 'replogle_qwen_metrics.py', 'replogle_qwen.py', 'run_replogle_qwen_eval.sh', 'submit_replogle_control_eval.sh']
METRICS = {
    "W2": ("distribution", "w2_pca30"), "MMD2": ("distribution", "mmd2_rbf_scldm"),
    "FD": ("distribution", "fd_pca30"), "PDS": ("cell_eval", "discrimination_score_l1"),
    "PR-AUC": ("cell_eval", "pr_auc"), "Sp-sig": ("cell_eval", "de_spearman_sig"),
    "Sp-LFC": ("cell_eval", "de_spearman_lfc_sig"), "ovl@N": ("cell_eval", "overlap_at_n"),
    "P-delta": ("cell_eval", "pearson_delta")}


def prepare_plan():
    check_inputs()
    store = CellStore()
    EVAL_ART.mkdir(parents=True, exist_ok=True)
    lines = np.array([r["cell_line"] for r in store.conditions])[store.condition_rows]
    reference = np.load(ART / "control_reference.npy")
    controls = {}
    for line in ["K562", "Jurkat", "RPE1", "HepG2"]:
        selected = np.flatnonzero(reference & (lines == line))
        if len(selected) > 5000:
            selected = np.sort(np.random.default_rng(420).choice(selected, 5000, replace=False))
        controls[line] = selected
        if line == "HepG2":
            assert len(selected) == 2502 and not store.support[selected].any()
    cases = []
    for split, mask in [("combination_validation", store.fold == 1), ("combination_test", store.fold == 2),
                        ("hepg2_paper", store.fold == 3), ("hepg2_all", np.isin(store.fold, [3, 5]))]:
        for line in sorted(set(lines[mask])):
            rows = np.flatnonzero(mask & (lines == line))
            key = f"{split}__{line}"
            np.save(EVAL_ART / f"{key}_targets.npy", rows)
            np.save(EVAL_ART / f"{key}_controls.npy", controls[line])
            cases.append({"key": key, "split": split, "cell_line": line, "target_cells": len(rows),
                          "conditions": len(np.unique(store.condition_rows[rows])),
                          "reference_cells": len(controls[line])})
    union = np.unique(np.concatenate([np.load(EVAL_ART / f"{c['key']}_{kind}.npy")
                                    for c in cases for kind in ["targets", "controls"]]))
    np.save(EVAL_ART / "generation_rows.npy", union)
    models = {}
    for record in read_json(ART / "active_submissions.json"):
        run = ROOT / "outputs/replogle_llm_control" / ART.name / record["variant"] / record["run_id"] / "formal"
        done = read_json(run / "completion.json")
        assert done["status"] == "complete"
        model = run / "final"
        assert sha(model / "model.safetensors") == sha(Path(done["best_checkpoint"]) / "model.safetensors")
        models[record["variant"]] = {"model": str(model), "selected_checkpoint": done["best_checkpoint"],
                                     "model_sha256": sha(model / "model.safetensors")}
    plan = {"cases": cases, "models": models, "unique_generation_cells": len(union), "seed": 420,
        "nb_depth": "Observed TARGET 2000-HVG raw-count row sum, unchanged author AR decode_observed protocol",
        "sampling": "one ancestral 64-code draw per target/reference row; temperature 1.0; no target FSQ in prompt",
        "controls": "same line/gem input support; source metric reference capped at 5000/line; HepG2 fixed disjoint 2502 reference cells",
        "gene_panel": "official 2000 genes; symbol-aligned target exclusion for PDS",
        "metrics": list(METRICS), "paper_subset_reuses_full_hepg2_predictions": True,
        "combination_aggregation": "per cell line, plus unweighted mean of the three cell-line scores; never pool cell lines in DE/PDS",
        "scope": "Stage-2 unseen HepG2; shared NB tokenizer already saw HepG2",
        "full_hepg2_includes_paper_subset": True, "target_cell_cap": 0,
        "distribution_max_samples": 5000, "pca_components": 30}
    save_json(EVAL_ART / "plan.json", plan)
    return store, plan


def verify_inputs():
    check_inputs()
    manifest = read_json(EVAL_ART / "ready_inputs.json")
    for name, expected in manifest["sha256"].items():
        _bootstrap.verify_input(name, expected)


class CachedRows(torch.utils.data.Dataset):
    """Expose the existing evaluator's dataset contract with original global row IDs."""
    def __init__(self, rows, counts, perturbations, cell_types, labels, genes):
        self._indices = np.asarray(rows, dtype=np.int64)
        self.counts = counts
        self._pert_indices, self._cell_type_indices = perturbations, cell_types
        self.unique_perts, self.unique_cell_types = labels["perturbations"], labels["cell_types"]
        self.pert_to_idx = {label: i for i, label in enumerate(self.unique_perts)}
        self._target_genes, self.num_genes = genes, len(genes)
        self.num_perts = len(self.unique_perts)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        row = int(self._indices[i])
        counts = self.counts[row]
        return {"counts": torch.from_numpy(counts), "library_size": float(counts.sum()),
                "pert_idx": int(self._pert_indices[row]), "cell_type_idx": int(self._cell_type_indices[row]),
                "row_id": row}


class CachedPredictions(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions

    def predict_batch(self, batch):
        rows = batch["row_id"].cpu().numpy()
        return torch.from_numpy(self.predictions[rows].copy()).to(batch["counts"].device)


def nine_metrics(summary):
    values = {}
    for label, (section, name) in METRICS.items():
        value = summary[section][name]
        values[label] = value.get("mean", value.get("value")) if isinstance(value, dict) else value
    return values
