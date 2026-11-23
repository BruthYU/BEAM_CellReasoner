"""CPU-only preparation and acceptance for the original-split 1-control experiment."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import json
import shutil

import numpy as np
import torch

from replogle_llm_common import (ROOT, DEFAULT_ART, read_json, save_json, sha,
    prepare_tokenizer, validation_indices, CellCollator, condition_prefixes)
from replogle_control_common import ART as AB_ART, CellStore
from replogle_control_baseline import (ART, CONFIG, OUTPUT_ROOT, NAME, FILES,
    BaselineControlDataset, runtime_config, safe_output, verify_inputs, make_training_arguments)


def copy_exact(source, dest):
    if dest.exists():
        assert sha(dest) == sha(source), f"Frozen copy differs: {dest}"
    else:
        shutil.copyfile(source, dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if args.verify_only or (ART / "ready_inputs.json").exists():
        verify_inputs()
        print("Frozen original-split 1-control inputs verified; no job submitted.")
        return
    torch.set_num_threads(2)
    ART.mkdir(parents=True, exist_ok=True)
    # Import the actual baseline files, not a new random split with matching counts.
    for name in ["cell_fold.npy", "condition_fold.npy", "split.json", "internal_validation_split.json"]:
        copy_exact(DEFAULT_ART / name, ART / name)
    # Reuse the previously raw-data-validated, globally aligned technical-group metadata.
    ab_manifest = read_json(AB_ART / "ready_inputs.json")["sha256"]
    for name in ["gem_group.npy", "gem_groups.json"]:
        assert sha(AB_ART / name) == ab_manifest[str((AB_ART / name).relative_to(ROOT))]
        copy_exact(AB_ART / name, ART / name)
    cache = DEFAULT_ART / "cache"
    cache_info = read_json(cache / "completion.json")
    assert cache_info["status"] == "complete"
    for name in ["token_ids", "condition_rows"]:
        assert sha(cache / f"{name}.npy") == cache_info[f"{name}_sha256"]
    source = read_json(DEFAULT_ART / "sources.json")
    initial = DEFAULT_ART / "initial_text_model"
    assert sha(initial / "model.safetensors") == read_json(initial / "export.json")["model_sha256"]
    sources = {**source, "nb_cache": str(cache.relative_to(ROOT)),
        "initial_model": str(initial.relative_to(ROOT)),
        "initial_model_sha256": sha(initial / "model.safetensors"), "uses_existing_sft_weights": False,
        "baseline_artifacts": str(DEFAULT_ART.relative_to(ROOT)),
        "gem_metadata_source": str(AB_ART.relative_to(ROOT))}
    save_json(ART / "sources.json", sources)
    conditions_path = ROOT / source["text_conditions"] / "conditions.jsonl"
    conditions = [json.loads(x) for x in conditions_path.read_text().splitlines()]
    assert sha(conditions_path) == read_json(DEFAULT_ART / "split.json")["conditions_sha256"]
    cr = np.load(cache / "condition_rows.npy")
    fold = np.load(ART / "cell_fold.npy")
    control = np.array([r["is_control"] for r in conditions])[cr]
    assert (fold[control] == 0).all()
    np.save(ART / "control_support.npy", control, allow_pickle=False)
    monitor = validation_indices(DEFAULT_ART, 20)
    assert len(monitor) == 4000
    np.save(ART / "validation_monitor.npy", monitor, allow_pickle=False)
    counts = {label: {"cells": int((fold == i).sum()),
        "controls": int(((fold == i) & control).sum()),
        "perturbed": int(((fold == i) & ~control).sum()),
        "conditions": len(np.unique(cr[fold == i]))}
        for i, label in enumerate(["train", "validation", "test"])}
    assert [counts[k]["cells"] for k in counts] == [924460, 14091, 28177]
    assert counts["train"]["controls"] == 39165
    base_cfg = read_json(ROOT / "outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1/formal/run_config.json")
    cfg = {k: base_cfg[k] for k in ["seed", "learning_rate", "weight_decay", "warmup_ratio",
        "epochs", "eval_steps", "logging_steps", "save_only_model"]}
    cfg.update(experiment=NAME, artifacts=str(ART.relative_to(ROOT)), output_root=str(OUTPUT_ROOT.relative_to(ROOT)),
        control_cells=1, eval_batch_size=16, save_steps=500, max_sequence_length=256,
        validation_cells_per_condition=20)
    cfg = runtime_config(cfg)
    save_json(CONFIG, cfg)
    store = CellStore(ART)
    lines = np.array([r["cell_line"] for r in conditions])[cr]
    for g, pool in store.pools.items():
        members = np.flatnonzero(store.gem == g)
        assert len(np.unique(lines[members])) == 1
        assert control[pool].all() and (fold[pool] == 0).all()
    tokenizer = prepare_tokenizer(initial)
    baseline_prefixes = condition_prefixes(tokenizer, conditions)
    examples, lengths = {}, {}
    datasets = {}
    for i, label in enumerate(counts):
        indices = np.flatnonzero(fold == i)
        ds = BaselineControlDataset(store, indices, tokenizer, training=i == 0)
        datasets[label] = ds
        assert ds.max_length <= 256
        lengths[label] = ds.max_length
        # One representative per condition + every control technical group.
        _, positions = np.unique(cr[indices], return_index=True)
        positions = list(positions)
        if i == 0:
            positions += [int(np.searchsorted(indices, pool[0])) for pool in store.pools.values()]
        for j in positions:
            row = int(indices[j]); c = ds.control_index(row)
            assert c != row and fold[c] == 0 and control[c]
            assert lines[c] == lines[row] and store.gem[c] == store.gem[row]
            prefix = ds.inference_prefix(j)
            visible = [t - ds.base for t in prefix if ds.base <= t < ds.base + 512]
            np.testing.assert_array_equal(visible, store.codes[c])
            ex = ds[j]
            assert ex["labels"][:len(prefix)] == [-100] * len(prefix)
            assert len(ex["labels"][len(prefix):]) == 66
            np.testing.assert_array_equal(np.array(ex["labels"][len(prefix):len(prefix)+64])-ds.base, store.codes[row])
        j = 0; row = int(indices[j]); c = ds.control_index(row); prompt = ds.inference_prefix(j)
        # Target codes cannot influence inference, including heldout targets.
        saved = store.codes[row].copy(); store.codes[row] = (saved + 1) % 512
        assert ds.inference_prefix(j) == prompt
        store.codes[row] = saved
        # Removing exactly the added user content restores the original chat prompt.
        content = conditions[int(cr[row])]["text"]
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert tokenizer.encode(rendered, add_special_tokens=False) + [ds.start] == baseline_prefixes[int(cr[row])]
        if i:
            ds.set_epoch(8); assert ds.inference_prefix(j) == prompt
        else:
            probes = indices[:200]
            before = [ds.control_index(int(r)) for r in probes]
            ds.set_epoch(1); after = [ds.control_index(int(r)) for r in probes]
            assert sum(a != b for a,b in zip(before, after)) > 180
            ds.set_epoch(0)
            assert before == [ds.control_index(int(r)) for r in probes]
        examples[label] = {"target_row": row, "control_row": c, "condition": content,
            "prompt": tokenizer.decode(prompt, skip_special_tokens=False),
            "target": tokenizer.decode(ds[j]["input_ids"][-66:], skip_special_tokens=False),
            "note": "Supervised observed target, not generated output"}
    np.testing.assert_array_equal(monitor, validation_indices(DEFAULT_ART, 20))
    assert (fold[monitor] == 1).all()
    # Check the real collator and loss path on CPU; full-vs-cropped logits must agree.
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    tiny = Qwen3_5TextConfig(vocab_size=601, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, layer_types=["full_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, max_position_embeddings=256, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .25,
                         "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    model = Qwen3_5ForCausalLM(tiny).eval()
    batch = CellCollator(tokenizer)([datasets["train"][0], datasets["validation"][0]])
    ids = batch["input_ids"] % 601
    labels = batch["labels"].clone(); labels[labels != -100] %= 601
    full = torch.full_like(ids, -100); full[:, -67:] = labels
    full_loss = model(input_ids=ids, attention_mask=batch["attention_mask"], labels=full, use_cache=False).loss
    loss = model(input_ids=ids, attention_mask=batch["attention_mask"], labels=labels, logits_to_keep=67, use_cache=False).loss
    torch.testing.assert_close(full_loss, loss, rtol=1e-6, atol=1e-6)
    loss.backward(); assert torch.isfinite(model.get_input_embeddings().weight.grad).all()
    for gpus, batch_size, global_batch in [(8,16,128), (8,8,128), (4,32,128), (8,32,256)]:
        for phase in ["smoke", "benchmark", "formal"]:
            c = runtime_config(cfg, gpus=gpus, batch_size=batch_size, global_batch=global_batch, phase=phase)
            ta = make_training_arguments(c, ART / "cpu_argument_check", cpu_check=True)
            assert ta.per_device_train_batch_size * ta.gradient_accumulation_steps * gpus == global_batch
            assert ta.save_steps == ta.eval_steps
            assert ta.activation_checkpointing is True
    off = make_training_arguments(runtime_config(cfg, gradient_checkpointing=False),
                                  ART / "cpu_argument_check", cpu_check=True)
    assert off.activation_checkpointing is False and off.gradient_checkpointing is False
    try:
        runtime_config(cfg, gpus=8, batch_size=32, global_batch=128)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid batch combination accepted")
    for protected in [ROOT / "outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1/formal",
                      ROOT / "outputs/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-v1/B"]:
        try:
            safe_output(protected)
        except AssertionError:
            pass
        else:
            raise AssertionError("Old output accepted")
    report = {"status": "passed", "counts": counts, "validation_monitor_cells": len(monitor),
        "split_byte_identical_to_baseline": True, "control_pool_cells": int(control.sum()),
        "sequence_lengths": lengths, "completion_only_loss_check": "full/cropped CE match; finite backward",
        "control_sampling": "all four lines, train-only NT, same line/gem, exclude self, epoch-resampled",
        "target_leakage_check": "changing target codes does not change prompt",
        "gpus": "not run; smoke and benchmark pending; no job submitted"}
    save_json(ART / "cpu_preflight.json", report)
    save_json(ART / "case_examples.json", examples)
    save_json(ART / "evaluation_plan.json", {"automatic_evaluation": "Opt-in via submit_replogle_qwen.sh --preset original-control1 --eval-after-training",
        "validation_cells": 14091, "test_cells": 28177,
        "reference_control_policy": "Original baseline evaluator: real HepG2 training NT, max_ctrl_cells=5000; do not use A/B half-pool",
        "input_control_policy": "Same line/gem, from 39165 baseline training NT, may overlap metric reference pool",
        "nb_depth": "Original observed target 2000-HVG total via decode_observed",
        "metrics": "Same nine original metrics, PDS gene-symbol aligned",
        "sampling": "one independent 64-code draw per target row, temperature=1, seed=420+global_batch_id",
        "population_evaluation_adapter": "8-way sharded generation, merged populations, unchanged original nine metrics"})
    from check_replogle_control_baseline_eval import run_checks
    run_checks()
    files = list(ART.glob("*.json")) + list(ART.glob("*.npy")) + [CONFIG]
    files += [_bootstrap.script_path(name) for name in FILES]
    files += list(initial.glob("*")) + list(cache.glob("*.npy"))
    files += [conditions_path, cache / "completion.json", DEFAULT_ART / "sources.json",
              DEFAULT_ART / "cell_fold.npy", DEFAULT_ART / "condition_fold.npy", DEFAULT_ART / "split.json",
              DEFAULT_ART / "internal_validation_split.json", ROOT / "assets/scldm_replogle_test.json",
              ROOT / "outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1/formal/run_config.json",
              AB_ART / "gem_group.npy", AB_ART / "gem_groups.json"]
    files += [ROOT / source["vae_run"] / "config.yaml",
              ROOT / source["vae_run"] / "checkpoints" / source["checkpoint"] / "model.safetensors",
              ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt",
              ROOT / "assets/replogle_scldm2k_gene_order.symbols.txt"]
    files += [ROOT / "src/sc-evae" / n for n in ["models/vae.py", "models/output_heads.py",
        "training/data_loader.py", "training/train_utils.py", "metrics/state_metrics.py",
        "metrics/distribution.py", "metrics/de_table.py"]]
    save_json(ART / "ready_inputs.json", {"status": "cpu_ready", "sha256": {
        str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files)) if f.is_file()}})
    verify_inputs()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
