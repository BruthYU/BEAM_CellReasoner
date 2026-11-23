"""Prepare an isolated A/B split and metadata on CPU; never submit a job."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "qwen35-08b-fsq-nb-hepg2-v1"
ART = ROOT / "artifacts/replogle_llm_control" / NAME
CONFIG = ROOT / "config/local/replogle_llm_control" / NAME
OLD = ROOT / "artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(16 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def frozen_json(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == content, f"Frozen file differs: {path}; use a new experiment ID"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def frozen_array(path, value):
    if path.exists():
        np.testing.assert_array_equal(np.load(path), value)
    else:
        np.save(path, value, allow_pickle=False)


def obs_column(obs, name):
    entry = obs[name]
    if isinstance(entry, h5py.Group):
        codes = entry["codes"][:]
        assert (codes >= 0).all()
        return entry["categories"].asstr()[:][codes]
    values = entry[:]
    if values.dtype.kind in "iu" and "__categories" in obs and name in obs["__categories"]:
        assert (values >= 0).all()
        return obs["__categories"][name].asstr()[:][values]
    return entry.asstr()[:] if values.dtype.kind in "OS" else values.astype(str)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    source = json.loads((OLD / "sources.json").read_text())
    text_dir = ROOT / source["text_conditions"]
    rows = [json.loads(s) for s in (text_dir / "conditions.jsonl").read_text().splitlines()]
    cr = np.load(OLD / "cache/condition_rows.npy")
    control = np.array([r["is_control"] for r in rows])[cr]
    source_conditions = [r for r in rows if r["cell_line"] != "HepG2"]
    remaining = Counter(r["dataset_label"] for r in source_conditions if not r["is_control"])
    # Greedy closest-to-budget selection, stratified by cell line and 10 size
    # strata. A gene must retain at least one source-line training condition.
    condition_fold = np.full(len(rows), 5, dtype=np.uint8)
    for r in source_conditions:
        condition_fold[r["condition_row"]] = 0
    rng = np.random.default_rng(420)
    selected = {1: [], 2: []}
    for ct in ["K562", "Jurkat", "RPE1"]:
        eligible = sorted([r for r in source_conditions if r["cell_line"] == ct and not r["is_control"]],
                          key=lambda r: (r["n_cells"], r["dataset_label"]))
        for stratum in np.array_split(np.arange(len(eligible)), 10):
            candidates = [eligible[int(j)] for j in rng.permutation(stratum)]
            budget = .05 * sum(eligible[int(j)]["n_cells"] for j in stratum)
            for fold_id in [1, 2]:
                used = 0
                for r in candidates:
                    i, gene, n = r["condition_row"], r["dataset_label"], r["n_cells"]
                    if n < 31 or condition_fold[i] != 0 or remaining[gene] < 2:
                        continue
                    if abs(used + n - budget) < abs(used - budget):
                        condition_fold[i] = fold_id
                        remaining[gene] -= 1
                        selected[fold_id].append(i)
                        used += n
    for r in rows:
        if r["cell_line"] == "HepG2" and r["split"] == "heldout":
            condition_fold[r["condition_row"]] = 3
    fold = condition_fold[cr]
    # Align metadata with the existing cache, verifying every cell's gene label.
    gem = np.empty(len(cr), dtype=np.int32)
    gem_labels = []
    manifest = json.loads((text_dir / "manifest.json").read_text())
    with np.load(text_dir / "cell_condition_index.npz") as ix:
        np.testing.assert_array_equal(cr, ix["condition_row"])
        for fidx, info in enumerate(manifest["files"]):
            mask = ix["source_file_index"] == fidx
            sr = ix["source_row"][mask]
            with h5py.File(ROOT / info["path"], "r") as h:
                actual_genes = obs_column(h["obs"], "gene")[sr]
                np.testing.assert_array_equal(actual_genes, np.array([rows[int(i)]["dataset_label"] for i in cr[mask]]))
                groups = obs_column(h["obs"], "gem_group")[sr]
                unique, inverse = np.unique(groups, return_inverse=True)
                gem[mask] = inverse + len(gem_labels)
                gem_labels.extend([f'{info["cell_line"]}:{g}' for g in unique])
    # HepG2 controls are inference-only: disjoint prompt support/reference pools.
    hepg2_controls = control & np.array([r["cell_line"] == "HepG2" for r in rows])[cr]
    fold[hepg2_controls] = 4
    support = np.zeros(len(cr), dtype=bool)
    reference = np.zeros(len(cr), dtype=bool)
    for group in np.unique(gem):
        indices = np.flatnonzero((gem == group) & control)
        assert len(indices) >= 2
        if hepg2_controls[indices[0]]:
            shuffled = rng.permutation(indices)
            support[shuffled[:len(indices) // 2]] = True
            reference[shuffled[len(indices) // 2:]] = True
        else:
            # Source-line controls all remain in training, as in the baseline.
            support[indices] = True
            reference[indices] = True
    assert not (support & reference & hepg2_controls).any()
    train_genes = {r["dataset_label"] for r in rows if condition_fold[r["condition_row"]] == 0}
    train_cts = {r["cell_line"] for r in rows if condition_fold[r["condition_row"]] == 0}
    for r in rows:
        k = condition_fold[r["condition_row"]]
        if k in [1, 2, 3]:
            assert r["dataset_label"] in train_genes
            assert (r["cell_line"] in train_cts) == (k != 3)
    assert "HepG2" not in train_cts and (fold == 3).sum() == 28177
    for name, array in [("cell_fold.npy", fold), ("condition_fold.npy", condition_fold),
                        ("gem_group.npy", gem), ("control_support.npy", support), ("control_reference.npy", reference)]:
        frozen_array(ART / name, array)
    # A fixed 20-cells/condition monitor; full group lists are retained for eval.
    monitor = []
    for k in sorted(selected[1]):
        candidates = np.flatnonzero(cr == k)
        monitor.extend(rng.choice(candidates, min(20, len(candidates)), replace=False).tolist())
    frozen_array(ART / "validation_monitor.npy", np.array(sorted(monitor), dtype=np.int64))
    fold_names = ["train", "combination_validation", "combination_test", "hepg2_test", "hepg2_controls", "unused_hepg2"]
    stats = {label: {"cells": int((fold == k).sum()), "controls": int(((fold == k) & control).sum()),
                     "perturbed": int(((fold == k) & ~control).sum()),
                     "conditions": int(len(np.unique(cr[fold == k])))} for k, label in enumerate(fold_names)}
    frozen_json(ART / "split.json", {"seed": 420, "fold_codes": dict(enumerate(fold_names)), "counts": stats,
        "source_lines": sorted(train_cts), "heldout_line": "HepG2", "validation_monitor_cells": len(monitor),
        "policy": "Whole condition holdout, approximately 5% source perturbed cells each for validation/test; stratified by line and group size; retain target in another training line",
        "control_policy": "Source controls all train; HepG2 controls inference-only, half prompt support/half metric reference per gem_group",
        "hepg2_support_cells": int((support & hepg2_controls).sum()),
        "hepg2_reference_cells": int((reference & hepg2_controls).sum()),
        "scope": "Stage-2 unseen cell line only: frozen NB tokenizer previously saw HepG2",
        "official_test_split_sha256": digest(ROOT / "assets/scldm_replogle_test.json")})
    for k in [1, 2, 3]:
        frozen_json(ART / (fold_names[k] + "_conditions.json"),
                    [r for r in rows if condition_fold[r["condition_row"]] == k])
    frozen_json(ART / "gem_groups.json", gem_labels)
    export = OLD / "initial_text_model"
    assert digest(export / "model.safetensors") == json.loads((export / "export.json").read_text())["model_sha256"]
    cache_info = json.loads((OLD / "cache/completion.json").read_text())
    assert digest(OLD / "cache/token_ids.npy") == cache_info["token_ids_sha256"]
    sources = {**source, "nb_cache": str((OLD / "cache").relative_to(ROOT)),
               "initial_model": str(export.relative_to(ROOT)), "initial_model_sha256": digest(export / "model.safetensors"),
               "uses_existing_sft_weights": False,
               "frozen_tokenizer_scope": "Shared pretrained NB tokenizer; HepG2 not unseen to Stage 1"}
    frozen_json(ART / "sources.json", sources)
    # Record options without changing the existing evaluator or choosing a depth
    # policy implicitly. Training is independent of NB decoding depth.
    frozen_json(ART / "evaluation_plan.json", {"automatic_evaluation": False,
        "splits": ["combination_validation", "combination_test", "hepg2_test"],
        "metrics": "Existing nine metrics, separately per split",
        "nb_decode_depth": "not_selected; choose control depth or source-training prior consistently for A/B before evaluation",
        "target_expression_as_model_input": False})
    for variant in ["A", "B"]:
        frozen_json(CONFIG / f"{variant}.json", {"experiment": NAME, "variant": variant,
            "artifacts": str(ART.relative_to(ROOT)), "output_root": f"outputs/replogle_llm_control/{NAME}/{variant}",
            "control_cells": 0 if variant == "A" else 1, "gpus": 4, "global_batch_size": 128,
            "seed": 420, "per_device_batch_size": 8, "eval_batch_size": 8,
            "learning_rate": 3e-5, "weight_decay": .01, "warmup_ratio": .03, "epochs": 3,
            "eval_steps": 500, "save_steps": 1000, "logging_steps": 20, "save_only_model": False,
            "max_sequence_length": 256, "charged_group": "beam_gpu",
            "node": f"gpu-lg-cmc-h-h200-{'0868' if variant == 'A' else '0862'}.host.h.pjlab.org.cn",
            "total_cpu_threads": 32, "cpu_threads_per_process": 8})
    # Snapshot old code/config and all baseline output metadata without writing
    # anywhere in those trees. Large weights are protected by size/mtime here.
    protected = {}
    roots = [OLD, ROOT / "outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1",
             ROOT / "config/local/replogle_llm/qwen35-08b-ar-fsq-nb-v1"]
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and not p.is_symlink() and not any(x in p.parts for x in ["__pycache__", "environment", "triton_cache"]):
                st = p.stat()
                protected[str(p.relative_to(ROOT))] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    frozen_json(ART / "baseline_snapshot.json", protected)
    print(json.dumps({"artifact_dir": str(ART), "counts": stats, "validation_monitor_cells": len(monitor)}, indent=2))


if __name__ == "__main__":
    main()
