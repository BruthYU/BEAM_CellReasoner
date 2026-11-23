"""CPU split/cache checks plus a synthetic round-trip through the unchanged evaluator."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml

from replogle_llm_common import ROOT, read_json, save_json, sha
from replogle_control_common import ART
from replogle_control_eval_common import EVAL_ART, SCRIPTS, METRICS, prepare_plan, verify_inputs, nine_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if args.verify_only:
        verify_inputs(); print("A/B evaluation inputs verified; no job submitted."); return
    import sc_evae.metrics.distribution as distribution
    assert Path(distribution.__file__).resolve() == (ROOT / "src/sc-evae/metrics/distribution.py").resolve(), "Use the same source_overlay as the evaluation worker"
    torch.set_num_threads(4)
    store, plan = prepare_plan()
    expected = {"combination_validation": (39123, 318), "combination_test": (39054, 320),
                "hepg2_paper": (28177, 372), "hepg2_all": (140497, 2393)}
    for split, (cells, groups) in expected.items():
        cases = [c for c in plan["cases"] if c["split"] == split]
        assert sum(c["target_cells"] for c in cases) == cells
        assert sum(c["conditions"] for c in cases) == groups
    union = np.load(EVAL_ART / "generation_rows.npy")
    assert np.all(np.diff(union) > 0)
    paper = np.load(EVAL_ART / "hepg2_paper__HepG2_targets.npy")
    all_hep = np.load(EVAL_ART / "hepg2_all__HepG2_targets.npy")
    assert np.isin(paper, all_hep).all()
    for case in plan["cases"]:
        targets = np.load(EVAL_ART / f"{case['key']}_targets.npy")
        refs = np.load(EVAL_ART / f"{case['key']}_controls.npy")
        assert not np.intersect1d(targets, refs).size
        assert not (store.fold[targets] == 0).any()
        assert all(store.conditions[int(c)]["cell_line"] == case["cell_line"] for c in store.condition_rows[refs])
        assert all(store.conditions[int(c)]["is_control"] for c in store.condition_rows[refs])
    assert sha(ROOT / "assets/scldm_replogle_test.json") == read_json(ART / "split.json")["official_test_split_sha256"]
    # Actual original evaluation path on identity predictions. This is a test
    # fixture only; it is never reported as an A/B experimental result.
    from replogle_qwen_legacy_eval import evaluate_case
    source = read_json(ART / "sources.json")
    config = yaml.safe_load((ROOT / source["vae_run"] / "config.yaml").read_text())
    config["_vae_run"] = source["vae_run"]
    deg_table = torch.load(ROOT / config["dataset"]["deg_table_path"], weights_only=False, map_location="cpu")
    target_conditions = np.unique(store.condition_rows[union[~np.array([r["is_control"] for r in store.conditions])[store.condition_rows[union]]]])
    assert all(store.conditions[int(i)]["dataset_label"] in deg_table for i in target_conditions)
    labels = read_json(ROOT / source["nb_cache"] / "labels.json")
    genes = (ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt").read_text().splitlines()
    symbols = (ROOT / "assets/replogle_scldm2k_gene_order.symbols.txt").read_text().splitlines()
    targets = sorted(set(symbols) & set(labels["perturbations"]))[:2]
    rng = np.random.default_rng(420)
    counts = rng.poisson(3, (192, 2000)).astype(np.float32)
    counts[64:128, :100] += 4
    counts[128:, 100:200] += 6
    pert = np.repeat([labels["perturbations"].index(x) for x in ["non-targeting"] + targets], 64)
    ctypes = np.full(192, labels["cell_types"].index("hepg2"))
    output = Path(tempfile.mkdtemp(prefix="cpu_evaluator_", dir=EVAL_ART))
    fixture = dict(key="synthetic_identity", split="cpu_fixture", cell_line="HepG2", target_cells=128,
                   conditions=2, reference_cells=64)
    result = evaluate_case(fixture, output, counts, counts.copy(), pert, ctypes, labels, genes,
                           np.arange(192), config, smoke=True)
    values = nine_metrics(result)
    assert all(np.isfinite(v) for v in values.values()), values
    assert values["W2"] < 1e-4 and abs(values["MMD2"]) < 1e-5 and abs(values["FD"]) < 1e-4
    assert values["PDS"] == 1 and values["P-delta"] > .9999
    report = {"status": "passed", "device": "cpu", "splits": expected,
              "unique_generation_cells": plan["unique_generation_cells"],
              "synthetic_identity_nine_metrics": values, "gpu_generation": "pending job startup",
              "unchanged_evaluator": "scripts/eval.py + original symbol alignment audit"}
    save_json(EVAL_ART / "cpu_preflight.json", report)
    files = [_bootstrap.script_path(name) for name in SCRIPTS + ["eval.py", "audit_replogle_stage2_pds.py"]]
    files += list(EVAL_ART.glob("*.npy")) + [EVAL_ART / "plan.json", EVAL_ART / "cpu_preflight.json"]
    files += [ROOT / "src/sc-evae" / n for n in ["models/vae.py", "models/output_heads.py",
        "training/data_loader.py", "training/train_utils.py", "metrics/state_metrics.py", "metrics/distribution.py"]]
    for info in plan["models"].values():
        files += [x for x in Path(info["model"]).iterdir() if x.is_file()]
    files += [ROOT / source["vae_run"] / "checkpoints" / source["checkpoint"] / "model.safetensors"]
    save_json(EVAL_ART / "ready_inputs.json", {"status": "cpu_ready", "sha256": {
        str(f.relative_to(ROOT)): sha(f) for f in files}})
    print(report, flush=True)


if __name__ == "__main__":
    main()
