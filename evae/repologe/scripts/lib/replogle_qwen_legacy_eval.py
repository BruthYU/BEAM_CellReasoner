"""Historical streaming and frozen-plan protocols; shared inference/metrics, immutable completed reports."""
# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import gc
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import draccus
import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import AutoTokenizer, set_seed

import eval as original_eval
from sc_evae.config.training import TrainConfig
from sc_evae.config.eval import EvalConfig
from sc_evae.training.data_loader import PerturbationDataset
from replogle_llm_common import DEFAULT_ART, condition_prefixes, ROOT, read_json, save_json, install_fast_delta, sha
from replogle_control_common import CellStore
from replogle_qwen import make_dataset
from replogle_qwen_inference import load_llm, load_nb_decoder, predict_cells
from replogle_control_eval_common import METRICS, nine_metrics


def control_protocol(variant):
    if variant == "C":
        from replogle_control_c_eval_common import EVAL_ART, EVAL_ROOT, verify_inputs
    elif variant in ("A", "B"):
        from replogle_control_eval_common import EVAL_ART, EVAL_ROOT, verify_inputs
    else:
        raise ValueError(variant)
    return EVAL_ART, EVAL_ROOT, verify_inputs


def generate(variant, output, batch_size):
    EVAL_ART, _, _ = control_protocol(variant)
    store = CellStore()
    plan = read_json(EVAL_ART / "plan.json")
    rows = np.load(EVAL_ART / "generation_rows.npy")
    cache_path = output / "generated_cells.h5"
    if (output / "generation_complete.json").exists():
        with h5py.File(cache_path, "r") as h:
            np.testing.assert_array_equal(h["global_rows"][:], rows)
            assert h.attrs["complete"] and h.attrs["variant"] == variant
        return
    assert not cache_path.exists(), "Partial generation exists; use a fresh run ID"
    set_seed(plan["seed"])
    install_fast_delta()
    with (ROOT / store.sources["vae_run"] / "config.yaml").open() as f:
        config = draccus.load(TrainConfig, f)
    config.dataset.pert_mapping_path = None
    assert not config.dataset.apply_normalize and not config.dataset.apply_log1p
    ds = PerturbationDataset(config=config.dataset)
    cache = ROOT / store.sources["nb_cache"]
    np.testing.assert_array_equal(ds._pert_indices, np.load(cache / "pert_idx.npy"))
    np.testing.assert_array_equal(ds._cell_type_indices, np.load(cache / "cell_type_idx.npy"))
    np.testing.assert_allclose(ds.counts[rows].sum(-1), np.load(cache / "hvg_library_size.npy")[rows], rtol=0, atol=0)
    model_dir = Path(plan["models"][variant]["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
    model = load_llm(model_dir, "cuda")
    vae = load_nb_decoder(ROOT / store.sources["vae_run"], "cuda")
    n = {"A": 0, "B": 1, "C": 8}[variant]
    prefixes = make_dataset(dict(preset="hepg2-" + variant.lower(), control_cells=n, seed=plan["seed"]), store, rows, tokenizer)
    started = time.monotonic()
    with h5py.File(cache_path, "w") as h:
        h.attrs.update(variant=variant, complete=False, seed=plan["seed"], nb_depth=plan["nb_depth"])
        h.create_dataset("global_rows", data=rows)
        h.create_dataset("condition_rows", data=store.condition_rows[rows])
        h.create_dataset("pert_idx", data=ds._pert_indices[rows])
        h.create_dataset("cell_type_idx", data=ds._cell_type_indices[rows])
        if variant == "C":
            h.create_dataset("control_input_rows", data=np.stack([prefixes.control_indices(int(r)) for r in rows]))
            h.attrs["control_cells"] = n
        else:
            h.create_dataset("control_input_row", data=np.array([prefixes.control_index(int(r)) if variant == "B" else -1 for r in rows]))
        codes_out = h.create_dataset("generated_codes", (len(rows), 64), dtype="uint16")
        pred_out = h.create_dataset("predicted_counts", (len(rows), 2000), dtype="float32", chunks=(64, 2000))
        true_out = h.create_dataset("true_counts", (len(rows), 2000), dtype="float32", chunks=(64, 2000))
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            counts = torch.from_numpy(ds.counts[rows[start:end]].copy()).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                codes, prediction = predict_cells(model, tokenizer, vae,
                    [prefixes.inference_prefix(i) for i in range(start, end)], counts)
            assert torch.isfinite(prediction).all() and (prediction >= 0).all()
            codes_out[start:end] = codes.cpu().numpy().astype(np.uint16)
            pred_out[start:end] = prediction.cpu().numpy()
            true_out[start:end] = counts.float().cpu().numpy()
            if start == 0 or end % (batch_size * 100) == 0 or end == len(rows):
                h.flush()
                save_json(output / "generation_progress.json", {"cells": end, "total": len(rows),
                    "elapsed_seconds": time.monotonic()-started, "cells_per_second": end/(time.monotonic()-started),
                    "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3})
                print(f"GENERATION {variant}: {end}/{len(rows)}", flush=True)
        h.attrs["complete"] = True
    save_json(output / "generation_complete.json", {"status": "complete", "cells": len(rows),
        "elapsed_seconds": time.monotonic()-started, "model": plan["models"][variant],
        "nb_depth": plan["nb_depth"], "generation_batch_size": batch_size,
        "gpu": torch.cuda.get_device_name(), "prompt_target_codes": False, "control_cells": n})
    del model, vae, ds, store, prefixes
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_case(*args, **kwargs):
    # Retain the historical A/B CSV-audit policy for this compatibility entry.
    from replogle_qwen_metrics import evaluate_case as shared_evaluate
    return shared_evaluate(*args, exact_pds=False, **kwargs)


def metrics(variant, output):
    EVAL_ART, _, _ = control_protocol(variant)
    plan = read_json(EVAL_ART / "plan.json")
    source = read_json(ROOT / "artifacts/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-v1/sources.json")
    labels = read_json(ROOT / source["nb_cache"] / "labels.json")
    genes = (ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt").read_text().splitlines()
    with h5py.File(output / "generated_cells.h5", "r") as h:
        assert h.attrs["complete"]
        rows, counts, predictions = h["global_rows"][:], h["true_counts"][:], h["predicted_counts"][:]
        pert, ctypes = h["pert_idx"][:], h["cell_type_idx"][:]
    config = yaml.safe_load((ROOT / source["vae_run"] / "config.yaml").read_text())
    config["_vae_run"] = source["vae_run"]
    reports = []
    for case in plan["cases"]:
        wanted = np.sort(np.concatenate([np.load(EVAL_ART / f"{case['key']}_{k}.npy") for k in ["targets", "controls"]]))
        selected = np.searchsorted(rows, wanted)
        np.testing.assert_array_equal(rows[selected], wanted)
        from replogle_qwen_metrics import evaluate_case as shared_evaluate
        summary = shared_evaluate(case, output, counts, predictions, pert, ctypes, labels, genes,
                                  selected, config, exact_pds=(variant == "C"))
        reports.append({**case, "nine_metrics": nine_metrics(summary)})
        save_json(output / "reports_progress.json", reports)
    aggregates = {}
    for split in ["combination_validation", "combination_test"]:
        group = [r for r in reports if r["split"] == split]
        aggregates[split] = {"aggregation": "unweighted cell-line mean; per-line scores retained",
            "nine_metrics": {m: float(np.mean([r["nine_metrics"][m] for r in group])) for m in METRICS}}
    save_json(output / "all_metrics.json", {"variant": variant, "reports": reports,
        "combination_macro_cell_line": aggregates, "protocol": plan})
    save_json(output / "completion.json", {"status": "complete", "variant": variant,
        "reports": len(reports), "all_nine_metrics_reported": True, "source_models_unchanged": True})


def run_control(args):
    _, output_root, verify = control_protocol(args.variant)
    output = args.output.resolve()
    assert output.is_relative_to(output_root.resolve()) and output != output_root.resolve()
    assert args.batch_size > 0
    verify()
    assert not (output / "completion.json").exists(), "Completed evaluation is immutable"
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(16)
    generate(args.variant, output, args.batch_size)
    metrics(args.variant, output)


class LLMAdapter(nn.Module):
    def __init__(self, model_dir, art):
        super().__init__()
        source = read_json(art / "sources.json")
        self.llm = load_llm(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
        self.vae = load_nb_decoder(ROOT / source["vae_run"])
        self.conditions = [json.loads(line) for line in (ROOT / source["text_conditions"] / "conditions.jsonl").read_text().splitlines()]
        self.prefixes = condition_prefixes(self.tokenizer, self.conditions)
        labels = read_json(art / "cache/labels.json")
        self.lookup = {(labels["cell_types"].index(row["source_cell_type"]),
                        labels["perturbations"].index(row["dataset_label"])): row["condition_row"]
                       for row in self.conditions}

    @torch.inference_mode()
    def predict_batch(self, batch):
        condition_rows = [self.lookup[(int(c), int(p))] for c, p in
                          zip(batch["cell_type_idx"].cpu(), batch["pert_idx"].cpu())]
        _, predicted = predict_cells(self.llm, self.tokenizer, self.vae,
            [self.prefixes[i] for i in condition_rows], batch["counts"])
        return predicted


def run_text(args):
    art, output = args.artifacts.resolve(), args.output_dir.resolve()
    assert args.batch_size > 0
    assert not (output / "summary_metrics.json").exists(), "Completed evaluation is immutable"
    output.mkdir(parents=True, exist_ok=True)
    install_fast_delta()
    set_seed(420)
    source = read_json(art / "sources.json")
    cfg = yaml.safe_load((ROOT / source["vae_run"] / "config.yaml").read_text())
    # This config is an evaluator bridge, not a claim that the LLM is the old AR.
    # The existing evaluator reads it for preprocessing, gene order and split.
    cfg["model"] = {"type": "autoregressive", "vae_path": source["vae_run"],
                    "do_pert_cond": False, "do_celltype_cond": False,
                    "transformer": {"type": "dit"}, "output_head": {"type": "cross_entropy"}}
    cfg["dataset"]["pert_mapping_path"] = None
    assert (ROOT / cfg["dataset"]["deg_table_path"]).is_file(), "Use the NB run's original shared DEG cache"
    split_path = {"quick": art / "quick_validation_split.json", "validation": art / "internal_validation_split.json",
                  "test": ROOT / "assets/scldm_replogle_test.json"}[args.split]
    cfg["dataset"]["split_path"] = str(split_path)
    bridge = output / "eval_bridge"
    (bridge / "checkpoints/llm").mkdir(parents=True, exist_ok=True)
    (bridge / "checkpoints/latest").write_text("llm\n")
    (bridge / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    model_dir = args.model_dir.resolve()
    originals = {k: getattr(original_eval, k) for k in ("make_model", "_load_model_weights", "_generate_predictions")}
    original_eval.make_model = lambda config: LLMAdapter(model_dir, art)
    original_eval._load_model_weights = lambda *args, **kwargs: None  # Adapter loads the actual HF checkpoint above.
    original_eval._generate_predictions = lambda model, batch, num_sample_steps: model.predict_batch(batch)
    eval_cfg = EvalConfig(
        train_dir=str(bridge), output_dir=str(output), batch_size=args.batch_size, num_workers=0,
        device="cuda", max_cells_per_pert=64 if args.split == "quick" else 0,
        max_ctrl_cells=512 if args.split == "quick" else 5000,
    )
    try:
        original_eval.main.__wrapped__(eval_cfg)
    finally:
        for key, value in originals.items():
            setattr(original_eval, key, value)
    assert (output / "summary_metrics.json").is_file()
    save_json(output / "llm_evaluation_provenance.json", {
        "model_dir": str(model_dir), "model_config_sha256": sha(model_dir / "config.json"),
        "evaluator": "scripts/eval.py", "evaluator_sha256": sha(ROOT / "scripts/eval.py"),
        "split": args.split, "split_path": str(split_path), "split_sha256": sha(split_path),
        "sampling": "One ancestral 64-code draw per cell, text conditions only; no target tokens or expression enter LLM",
        "nb_depth": "Observed HVG row sum, exactly matching the existing AR evaluation protocol",
        "deployment_depth": "Use VAE.decode() and its training prior when observed target depth is unavailable",
        "existing_evaluation_code_modified": False,
    })
    # Reuse the existing Ensembl-to-symbol target-exclusion audit for PDS.
    import subprocess
    import sys
    # That helper expects a run directory containing eval_results/.
    audit_run = output / "pds_audit_run"
    audit_run.mkdir(exist_ok=True)
    link = audit_run / "eval_results"
    if not link.exists():
        link.symlink_to(output, target_is_directory=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/checks/audit_replogle_stage2_pds.py"),
                    "--run-dir", str(audit_run)], check=True)
