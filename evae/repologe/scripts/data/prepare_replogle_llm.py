"""Freeze an internal condition holdout before fitting the single-cell LLM prior."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ART = "artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
NB_RUN = "outputs/replogle_experiments/ar-fsq-nb-l64-k512-990fc9c87a-2gpu-0868/stage1"
LLM = "/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue/BEAM_CellReasoner/model/LLM/Qwen3.5-0.8B"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        assert path.read_text() == content, f"Refusing to change frozen experiment: {path}"
    else:
        path.write_text(content)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", default=DEFAULT_ART)
    p.add_argument("--validation-conditions", type=int, default=200)
    p.add_argument("--minimum-validation-cells", type=int, default=31)
    p.add_argument("--seed", type=int, default=420)
    args = p.parse_args()
    art = ROOT / args.artifacts
    art.mkdir(parents=True, exist_ok=True)
    conditions = ROOT / "artifacts/replogle_text_conditions/v1"
    rows = [json.loads(line) for line in (conditions / "conditions.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(args.seed)
    eligible = [row for row in rows if row["cell_line"] == "HepG2" and
                row["split"] == "train" and not row["is_control"] and
                row["n_cells"] >= args.minimum_validation_cells]
    eligible.sort(key=lambda row: (row["n_cells"], row["dataset_label"]))
    assert len(eligible) >= args.validation_conditions
    # Stratify by condition size, using metadata only. Small groups remain train.
    strata = np.array_split(np.arange(len(eligible)), 10)
    selected = []
    for i, stratum in enumerate(strata):
        n = args.validation_conditions // 10 + (i < args.validation_conditions % 10)
        selected.extend(eligible[j] for j in rng.choice(stratum, n, replace=False))
    selected.sort(key=lambda row: row["dataset_label"])
    val_rows = {row["condition_row"] for row in selected}
    test_rows = {row["condition_row"] for row in rows if row["split"] == "heldout"}
    assert not val_rows & test_rows
    condition_fold = np.zeros(len(rows), dtype=np.uint8)
    condition_fold[list(val_rows)] = 1
    condition_fold[list(test_rows)] = 2
    with np.load(conditions / "cell_condition_index.npz") as index:
        cell_condition = index["condition_row"]
        fold = condition_fold[cell_condition]
    assert len(fold) == 966728 and int((fold == 2).sum()) == 28177
    assert len(val_rows) == args.validation_conditions and len(test_rows) == 372
    for name, value in [("cell_fold.npy", fold), ("condition_fold.npy", condition_fold)]:
        path = art / name
        if path.exists():
            np.testing.assert_array_equal(np.load(path), value)
        else:
            np.save(path, value, allow_pickle=False)
    # Fixed quick generation subset, spanning the validation group-size range.
    ranked_val = sorted(selected, key=lambda row: (row["n_cells"], row["dataset_label"]))
    quick = [ranked_val[i] for i in np.linspace(0, len(ranked_val) - 1, 20, dtype=int)]
    record = {
        "status": "passed", "seed": args.seed, "fold_codes": {"train": 0, "validation": 1, "test": 2},
        "validation_strategy": "HepG2 condition holdout, stratified into 10 condition-size strata",
        "eligible_hepg2_training_conditions": len(eligible),
        "minimum_validation_cells": args.minimum_validation_cells,
        "minimum_validation_cells_reason": "Match the outer test's minimum group size and allow population metrics",
        "train_cells": int((fold == 0).sum()), "validation_cells": int((fold == 1).sum()),
        "test_cells": int((fold == 2).sum()), "train_conditions": int((condition_fold == 0).sum()),
        "validation_conditions": len(val_rows), "test_conditions": len(test_rows),
        "validation_labels": [row["dataset_label"] for row in selected],
        "validation_condition_rows": sorted(val_rows),
        "validation_group_counts": {row["dataset_label"]: row["n_cells"] for row in selected},
        "quick_validation_labels": [row["dataset_label"] for row in quick],
        "outer_split": "assets/scldm_replogle_test.json", "outer_split_sha256": sha(ROOT / "assets/scldm_replogle_test.json"),
        "conditions_sha256": sha(conditions / "conditions.jsonl"),
        "controls": "All controls stay in train; no cells from these heldout condition groups enter SFT",
        "other_cell_lines": "The same target genes in other cell lines remain train, matching the outer split design",
        "scope": "Stage-2 validation with a fixed tokenizer previously fit on the original training set",
    }
    save_json(art / "split.json", record)
    save_json(art / "internal_validation_split.json", {"cell_type": ["hepg2"], "pert": record["validation_labels"]})
    save_json(art / "quick_validation_split.json", {"cell_type": ["hepg2"], "pert": record["quick_validation_labels"]})
    model_path = ROOT / NB_RUN
    checkpoint = (model_path / "checkpoints/latest").read_text().strip()
    assert checkpoint == "step_0075000_final"
    source = {
        "head": "nb", "num_latent_tokens": 64, "codebook_size": 512,
        "vae_run": NB_RUN, "checkpoint": checkpoint,
        "checkpoint_sha256": sha(model_path / "checkpoints" / checkpoint / "model.safetensors"),
        "config_sha256": sha(model_path / "config.yaml"),
        "model_source": LLM, "model_revision": json.loads((Path(LLM) / "LOCAL_SOURCE.json").read_text())["revision"],
        "text_conditions": "artifacts/replogle_text_conditions/v1", "seed": args.seed,
    }
    save_json(art / "sources.json", source)
    print(json.dumps({k: v for k, v in record.items() if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
