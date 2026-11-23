"""Shared cached-population bridge to the author's nine metric implementations."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap

import subprocess
import sys
import numpy as np
import yaml
import eval as original_eval
from sc_evae.metrics.state_metrics import aggregate_per_pert
from sc_evae.config.eval import EvalConfig
from replogle_llm_common import ROOT, read_json, save_json, sha
from replogle_control_eval_common import CachedRows, CachedPredictions, nine_metrics, METRICS

def evaluate_case(case, output, counts, predictions, pert, ctypes, labels, genes, selected, vae_config,
                  *, smoke=False, exact_pds=True):
    report_dir = output / "reports" / case["key"]
    result_dir = report_dir / "eval_results"
    if (report_dir / "completion.json").exists():
        return read_json(result_dir / "summary_metrics_symbol_aligned.json")
    result_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vae_config)
    config["dataset"] = dict(vae_config["dataset"], pert_mapping_path=None, split_path=None)
    config["model"] = {"type": "autoregressive", "vae_path": vae_config.get("_vae_run"),
        "do_pert_cond": False, "do_celltype_cond": False, "transformer": {"type": "dit"},
        "output_head": {"type": "cross_entropy"}}
    config.pop("_vae_run", None)
    bridge = report_dir / "eval_bridge"
    (bridge / "checkpoints/cached").mkdir(parents=True, exist_ok=True)
    (bridge / "checkpoints/latest").write_text("cached\n")
    (bridge / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    ds = CachedRows(selected, counts, pert, ctypes, labels, genes)
    assert len(np.unique(ctypes[selected])) == 1
    originals = {k: getattr(original_eval, k) for k in ["PerturbationDataset", "resolve_train_val_split",
                 "make_model", "_load_model_weights", "_generate_predictions"]}
    original_eval.PerturbationDataset = lambda **kwargs: ds
    original_eval.resolve_train_val_split = lambda *args: (ds, ds)
    original_eval.make_model = lambda config: CachedPredictions(predictions)
    original_eval._load_model_weights = lambda *args: None
    original_eval._generate_predictions = lambda model, batch, num_sample_steps: model.predict_batch(batch)
    pds_inputs = {}
    pds_function = original_eval.discrimination_score_l1
    def capture_pds(real, predicted, control, labels, **kwargs):
        pds_inputs.update(real=real.copy(), predicted=predicted.copy(), control=control.copy(), labels=list(labels), kwargs=kwargs)
        return pds_function(real, predicted, control, labels, **kwargs)
    original_eval.discrimination_score_l1 = capture_pds
    try:
        # Only the input/cache adapter changes. All metric implementations remain original.
        original_eval.main.__wrapped__(EvalConfig(train_dir=str(bridge), output_dir=str(result_dir),
            batch_size=256, num_workers=0, device="cpu", max_cells_per_pert=0, max_ctrl_cells=0,
            de_table_path=str(output / "real_de_cache" / (case["key"]+".csv.gz")),
            distribution_max_samples=128 if smoke else 5000, pca_components=30))
    finally:
        for k, v in originals.items(): setattr(original_eval, k, v)
        original_eval.discrimination_score_l1 = pds_function
    if exact_pds:
        assert pds_inputs, "Original evaluator did not compute PDS"
        symbols = (ROOT / "assets/replogle_scldm2k_gene_order.symbols.txt").read_text().splitlines()
        saved = read_json(result_dir / "state_per_pert.json")
        # Exact arrays from the original evaluator: no float64 DE-table roundtrip.
        kwargs = dict(pds_inputs["kwargs"], gene_names=symbols)
        corrected = pds_function(pds_inputs["real"], pds_inputs["predicted"], pds_inputs["control"], pds_inputs["labels"], **kwargs)
        np.savez(result_dir / "pds_exact_inputs.npz", real=pds_inputs["real"], predicted=pds_inputs["predicted"], control=pds_inputs["control"], labels=pds_inputs["labels"])
        audit = {"method": "Original PDS function with exact evaluator float32 pseudobulks and symbol-aligned target exclusion",
                 "symbol_aligned_target_excluded": aggregate_per_pert(corrected), "per_pert_symbol_aligned_pds": corrected,
                 "original": aggregate_per_pert({k: v["discrimination_score_l1"] for k, v in saved.items()}),
                 "exact_inputs_sha256": sha(result_dir / "pds_exact_inputs.npz")}
        save_json(result_dir / "pds_target_gene_audit.json", audit)
        summary = read_json(result_dir / "summary_metrics.json")
        summary["cell_eval"]["discrimination_score_l1"] = audit["symbol_aligned_target_excluded"]
        save_json(result_dir / "summary_metrics_symbol_aligned.json", summary)
    else:
        subprocess.run([sys.executable, str(ROOT / "scripts/checks/audit_replogle_stage2_pds.py"),
                        "--run-dir", str(report_dir)], check=True)
    per_pert = read_json(result_dir / "per_pert_metrics.json")
    assert len(per_pert) == case["conditions"], "Evaluator skipped a requested perturbation"
    assert sum(r["n_cells"] for r in per_pert.values()) == case["target_cells"], "Target cells were omitted"
    summary = read_json(result_dir / "summary_metrics_symbol_aligned.json")
    values = nine_metrics(summary)
    assert set(values) == set(METRICS)
    save_json(report_dir / "completion.json", {"status": "complete", **case, "nine_metrics": values,
        "eval_code_sha256": sha(ROOT / "scripts/eval.py"), "no_gene_or_cell_line_pooling": True})
    return summary
