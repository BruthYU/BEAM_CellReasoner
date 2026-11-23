"""CPU checks for variable-size control sets; never submit jobs or alter A/B."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse

import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from replogle_llm_common import ROOT, CellCollator, prepare_tokenizer, read_json, save_json, sha
from replogle_control_common import ControlCellDataset, check_inputs as check_ab_inputs
from check_replogle_control_ab import baseline_unchanged
from replogle_control_c import (ART, CONFIG, SPLIT_ART, CellStore, ControlSetDataset,
    runtime_config, safe_output, check_inputs, positive_int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-cells-nums", type=positive_int, default=None)
    parser.add_argument("--gpus", type=int, choices=[2, 4], default=4)
    parser.add_argument("--per-device-batch-size", type=positive_int, default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cfg = runtime_config(read_json(CONFIG), args.gpus, "formal", args.control_cells_nums,
                         args.per_device_batch_size)
    baseline_unchanged()
    if args.verify_only:
        check_inputs()
        print(f"C inputs verified; N={cfg['control_cells']}, global batch={cfg['global_batch_size']}, "
              f"accumulation={cfg['gradient_accumulation_steps']}; no job submitted.")
        return
    check_ab_inputs()
    torch.set_num_threads(2)
    store = CellStore()
    tok = prepare_tokenizer(ROOT / store.sources["initial_model"])
    # Include every source technical group and non-targeting self-exclusion cases.
    indices = np.array([np.flatnonzero((store.fold == 0) & (store.gem == g))[0]
                       for g in np.unique(store.gem[store.fold == 0])], dtype=np.int64)
    controls = np.flatnonzero(store.support & (store.fold == 0))[:8]
    indices = np.unique(np.concatenate([indices, controls]))
    lengths, examples, batches = {}, {}, []
    reference = np.load(SPLIT_ART / "control_reference.npy")
    for n in sorted({1, 8, 32, cfg["control_cells"]}):
        ds = ControlSetDataset(store, indices, tok, n, training=True)
        for i, row in enumerate(indices):
            selected = ds.control_indices(int(row))
            available = store.pools[int(store.gem[row])]
            available = available[available != row]
            assert len(selected) == n and row not in selected
            assert store.support[selected].all() and (store.fold[selected] == 0).all()
            assert (store.gem[selected] == store.gem[row]).all()
            assert len(np.unique(selected)) == min(n, len(available))
            target_line = store.conditions[int(store.condition_rows[row])]["cell_line"]
            assert all(store.conditions[int(store.condition_rows[c])]["cell_line"] == target_line for c in selected)
        if n == 1:
            b = ControlCellDataset(store, indices, tok, "B", training=True)
            assert all(ds.control_indices(int(r))[0] == b.control_index(int(r)) for r in indices)
        before = [ds.control_indices(int(r)).tolist() for r in indices[:10]]
        ds.set_epoch(1)
        assert before != [ds.control_indices(int(r)).tolist() for r in indices[:10]]
        ds.set_epoch(0)
        assert before == [ds.control_indices(int(r)).tolist() for r in indices[:10]]
        example = ds[0]
        prefix = ds.inference_prefix(0)
        assert prefix.count(ds.start) == n + 1 and prefix.count(ds.end) == n
        assert sum(v != -100 for v in example["labels"]) == 66
        assert example["labels"][:len(prefix)] == [-100] * len(prefix)
        assert example["input_ids"][len(prefix):len(prefix)+64] == (store.codes[indices[0]].astype(np.int64)+ds.base).tolist()
        old = store.codes[indices[0]].copy()
        try:
            store.codes[indices[0]] = (old + 1) % 512
            assert prefix == ds.inference_prefix(0)
        finally:
            store.codes[indices[0]] = old
        # Deliberately use the smallest HepG2 pool to test the shortage policy.
        hep = np.flatnonzero(store.fold == 3)
        row = min(hep, key=lambda r: len(store.pools[int(store.gem[r])]))
        test = ControlSetDataset(store, [row], tok, n)
        selected = test.control_indices(int(row))
        assert (store.fold[selected] == 4).all() and not reference[selected].any()
        assert len(np.unique(selected)) == min(n, len(store.pools[int(store.gem[row])]))
        val = ControlSetDataset(store, np.load(SPLIT_ART / "validation_monitor.npy")[:2], tok, n)
        fixed = val.inference_prefix(0)
        val.set_epoch(8)
        assert val.inference_prefix(0) == fixed
        lengths[str(n)] = {"train_probe": ds.max_length, "validation_probe": val.max_length,
                           "hepg2_probe": test.max_length}
        budget = runtime_config(read_json(CONFIG), args.gpus, "formal", n)["max_sequence_length"]
        assert max(lengths[str(n)].values()) <= budget
        if n in [1, 8]:
            batches.append(CellCollator(tok)([ds[0], ds[-1]]))
        examples[str(n)] = {"target_row": int(indices[0]), "control_rows": ds.control_indices(int(indices[0])).tolist(),
                           "input": tok.decode(prefix), "output": tok.decode(example["input_ids"][len(prefix):])}
    # Same completion-only objective at long C prompts: full/cropped CE agree.
    tiny = Qwen3_5TextConfig(vocab_size=601, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, layer_types=["full_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, max_position_embeddings=4096, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .25,
                         "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    model = Qwen3_5ForCausalLM(tiny).eval()
    for batch in batches:
        ids = batch["input_ids"] % 601
        labels = batch["labels"].clone()
        labels[labels != -100] %= 601
        full = torch.full_like(ids, -100)
        full[:, -67:] = labels
        a = model(input_ids=ids, attention_mask=batch["attention_mask"], labels=full, use_cache=False).loss
        b = model(input_ids=ids, attention_mask=batch["attention_mask"], labels=labels,
                  logits_to_keep=67, use_cache=False).loss
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        b.backward()
        assert torch.isfinite(model.get_input_embeddings().weight.grad).all()
        model.zero_grad(set_to_none=True)
    for g in [2, 4]:
        for phase in ["smoke", "benchmark", "formal"]:
            c = runtime_config(read_json(CONFIG), g, phase)
            assert g * c["per_device_batch_size"] * c["gradient_accumulation_steps"] == 128
    try:
        safe_output(ROOT / "outputs/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-v1/B")
    except AssertionError:
        pass
    else:
        raise AssertionError("C must not write into A/B")
    baseline_unchanged()
    report = {"status": "passed", "device": "cpu", "tested_N": sorted(map(int, lengths)),
              "sequence_lengths": lengths, "global_batch": 128,
              "checks": ["same line/gem", "self-exclusion", "no target codes in prompt",
                         "shortage with replacement", "HepG2 support/reference separation",
                         "epoch resampling; fixed validation", "66 supervised output tokens",
                         "full vs cropped loss and finite backward", "A/B inputs unchanged"],
              "gpu_checks": "not_run; smoke gate required before formal training"}
    save_json(ART / "cpu_preflight.json", report)
    save_json(ART / "case_examples.json", examples)
    files = [CONFIG, SPLIT_ART / "ready_inputs.json", ART / "cpu_preflight.json", ART / "case_examples.json"]
    files += [_bootstrap.script_path(f) for f in ["replogle_control_c.py", "train_replogle_control_c.py",
        "check_replogle_control_c.py", "submit_replogle_qwen.sh", "submit_replogle_qwen.py", "run_replogle_qwen_worker.sh", "train_replogle_qwen.py", "replogle_qwen.py", "replogle_qwen_callbacks.py", "run_replogle_control_c_worker.sh"]]
    save_json(ART / "ready_inputs.json", {"status": "cpu_ready", "sha256": {
        str(p.relative_to(ROOT)): sha(p) for p in files}})
    print(report)


if __name__ == "__main__":
    main()
