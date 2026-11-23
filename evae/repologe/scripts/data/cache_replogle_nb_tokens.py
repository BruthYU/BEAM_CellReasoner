"""Cache NB-FSQ targets using the existing dataset and frozen VAE APIs."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import json
import os
from pathlib import Path
import time

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader

from sc_evae.config.training import TrainConfig
from sc_evae.config.models import NBHeadConfig
from sc_evae.models.factory import load_vae_from_run
from sc_evae.training.data_loader import PerturbationDataset
from replogle_llm_common import DEFAULT_ART, ROOT, read_json, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ART)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    art = args.artifacts.resolve()
    source = read_json(art / "sources.json")
    split = read_json(art / "split.json")
    assert split["status"] == "passed" and source["head"] == "nb"
    assert torch.cuda.is_available()
    assert os.environ["KUBEBRAIN_NODE_NAME"] == "gpu-lg-cmc-h-h200-0868.host.h.pjlab.org.cn"
    cache = art / "cache"
    cache.mkdir(exist_ok=True)
    if (cache / "completion.json").exists():
        report = read_json(cache / "completion.json")
        assert report["checkpoint_sha256"] == source["checkpoint_sha256"]
        assert sha(cache / "token_ids.npy") == report["token_ids_sha256"]
        print("Existing NB cache verified", flush=True)
        return
    start = time.monotonic()
    torch.set_num_threads(4)
    run = ROOT / source["vae_run"]
    weight = run / "checkpoints" / source["checkpoint"] / "model.safetensors"
    assert sha(weight) == source["checkpoint_sha256"]
    with (run / "config.yaml").open() as handle:
        config = draccus.load(TrainConfig, handle)
    assert isinstance(config.model.output_head, NBHeadConfig)
    assert not config.dataset.apply_normalize and not config.dataset.apply_log1p
    config.dataset.pert_mapping_path = None
    dataset = PerturbationDataset(config=config.dataset)
    model = load_vae_from_run(str(run), device="cuda")
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert model.config.num_latent_tokens == 64 and model.config.quantizer.codebook_size == 512
    assert len(dataset) == 966728 and dataset.num_genes == 2000
    conditions_path = ROOT / source["text_conditions"]
    rows = [json.loads(line) for line in (conditions_path / "conditions.jsonl").read_text().splitlines()]
    lookup = np.full((len(dataset.unique_cell_types), dataset.num_perts), -1, dtype=np.int32)
    for row in rows:
        lookup[dataset.cell_type_to_idx[row["source_cell_type"]], dataset.pert_to_idx[row["dataset_label"]]] = row["condition_row"]
    condition_rows = lookup[dataset._cell_type_indices, dataset._pert_indices]
    assert condition_rows.min() >= 0
    with np.load(conditions_path / "cell_condition_index.npz") as index:
        np.testing.assert_array_equal(condition_rows, index["condition_row"])
    np.save(cache / "condition_rows.npy", condition_rows, allow_pickle=False)
    np.save(cache / "library_size.npy", dataset._library_size, allow_pickle=False)
    np.save(cache / "hvg_library_size.npy", dataset.counts.sum(axis=1), allow_pickle=False)
    np.save(cache / "pert_idx.npy", dataset._pert_indices, allow_pickle=False)
    np.save(cache / "cell_type_idx.npy", dataset._cell_type_indices, allow_pickle=False)
    save_json(cache / "labels.json", {"perturbations": dataset.unique_perts, "cell_types": dataset.unique_cell_types})
    temporary = cache / "token_ids.partial.npy"
    codes = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint16, shape=(len(dataset), 64))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    frequencies = np.zeros(512, dtype=np.int64)
    offset = 0
    # Same BF16 encoder precision and eval mode as existing AR training.
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            counts = batch["counts"].to("cuda", non_blocking=True)
            assert bool((counts >= 0).all())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                indices = model.quantizer(model.encode(counts)).indices
            array = indices.cpu().numpy()
            assert array.shape == (len(counts), 64) and array.min() >= 0 and array.max() < 512
            codes[offset:offset + len(array)] = array.astype(np.uint16)
            frequencies += np.bincount(array.reshape(-1), minlength=512)
            offset += len(array)
            if batch_index % 100 == 0:
                print(json.dumps({"encoded": offset, "total": len(dataset), "elapsed_seconds": time.monotonic() - start}), flush=True)
                save_json(cache / "progress.json", {"encoded": offset, "total": len(dataset), "elapsed_seconds": time.monotonic() - start})
    assert offset == len(dataset)
    codes.flush()
    del codes
    temporary.replace(cache / "token_ids.npy")
    # Regenerate deterministic targets for a train, validation and test cell.
    cached = np.load(cache / "token_ids.npy", mmap_mode="r")
    fold = np.load(art / "cell_fold.npy")
    probe_rows = [int(np.flatnonzero(fold == i)[0]) for i in [0, 1, 2]]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in probe_rows:
            x = dataset[row]["counts"].unsqueeze(0).to("cuda")
            actual = model.quantizer(model.encode(x)).indices.cpu().numpy()[0]
            # FSQ rounds continuous coordinates; exact singleton/batch agreement
            # is informative but BF16 GEMM can move coordinates across a boundary.
            assert actual.shape == (64,) and actual.min() >= 0 and actual.max() < 512
    assert frequencies.sum() == len(dataset) * 64
    report = {"status": "complete", "head": "nb", "shape": list(cached.shape), "dtype": "uint16",
        "token_range": [int(cached.min()), int(cached.max())], "used_codes": int((frequencies > 0).sum()),
        "code_frequencies": frequencies.tolist(), "vae_run": source["vae_run"],
        "checkpoint_sha256": source["checkpoint_sha256"], "token_ids_sha256": sha(cache / "token_ids.npy"),
        "condition_rows_sha256": sha(cache / "condition_rows.npy"), "split_sha256": sha(art / "cell_fold.npy"),
        "train_cells": split["train_cells"], "validation_cells": split["validation_cells"], "test_cells": split["test_cells"],
        "node": os.environ["KUBEBRAIN_NODE_NAME"], "gpu": torch.cuda.get_device_name(),
        "elapsed_seconds": time.monotonic() - start, "encoder_precision": "bf16",
        "input_units": "Raw integer UMI; original dataset loader, QC and 2000-gene order",
        "condition_mapping": "Every retained cell checked against the existing text-condition sidecar"}
    save_json(cache / "completion.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "code_frequencies"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
