"""CPU acceptance for real A/B examples, splits, resampling and completion loss."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, Trainer, TrainingArguments

from replogle_llm_common import (ROOT, CellCollator, FSQ_TOKENS, CELL_START, CELL_END,
    prepare_tokenizer, read_json, save_json, sha, training_arguments)
from replogle_control_common import (ART, CONFIG, CellStore, ControlCellDataset,
    SetControlEpoch, runtime_config, check_inputs, safe_output)


def baseline_unchanged():
    for name, before in read_json(ART / "baseline_snapshot.json").items():
        st = (ROOT / name).stat()
        assert {"size": st.st_size, "mtime_ns": st.st_mtime_ns} == before, f"Baseline changed: {name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        baseline_unchanged()
        check_inputs()
        print("Prepared inputs verified; baseline preserved; no job submitted.")
        return
    torch.set_num_threads(2)
    baseline_unchanged()
    store = CellStore()
    tok = prepare_tokenizer(ROOT / store.sources["initial_model"])
    folds = store.fold
    cr = store.condition_rows
    monitor = np.load(ART / "validation_monitor.npy")
    assert (folds[monitor] == 1).all()
    train_genes = {store.conditions[int(k)]["dataset_label"] for k in np.unique(cr[folds == 0])}
    train_lines = {store.conditions[int(k)]["cell_line"] for k in np.unique(cr[folds == 0])}
    assert train_lines == {"K562", "Jurkat", "RPE1"}
    for fold in [1, 2, 3]:
        for k in np.unique(cr[folds == fold]):
            row = store.conditions[int(k)]
            assert row["dataset_label"] in train_genes and not row["is_control"]
            assert (row["cell_line"] in train_lines) == (fold != 3)
            assert not np.any((cr == k) & (folds == 0))
    train_indices = np.flatnonzero(folds == 0)
    val_indices = monitor
    lengths, examples = {}, {}
    batches = []
    for variant in ["A", "B"]:
        ds = ControlCellDataset(store, train_indices, tok, variant, training=True)
        val = ControlCellDataset(store, val_indices, tok, variant)
        test = ControlCellDataset(store, np.flatnonzero(folds == 3), tok, variant)
        assert ds.max_length <= 256 and val.max_length <= 256 and test.max_length <= 256
        # Probe all source technical groups, plus targeted and control targets.
        probes = [int(np.flatnonzero(store.gem[train_indices] == g)[0]) for g in np.unique(store.gem[train_indices])]
        control_targets = [j for j in range(len(train_indices)) if store.support[train_indices[j]]][:40]
        probes += control_targets
        for j in probes:
            row = int(train_indices[j])
            prompt = ds.inference_prefix(j)
            example = ds[j]
            target = example["input_ids"][len(prompt):]
            assert len(target) == 66 and example["labels"][:len(prompt)] == [-100] * len(prompt)
            assert example["labels"][len(prompt):] == target
            np.testing.assert_array_equal(np.array(target[:64]) - ds.base, store.codes[row])
            if variant == "B":
                c = ds.control_index(row)
                assert folds[c] == 0 and c != row and store.support[c]
                assert store.conditions[int(cr[c])]["is_control"]
                assert store.conditions[int(cr[c])]["cell_line"] == store.conditions[int(cr[row])]["cell_line"]
                visible_codes = [x - ds.base for x in prompt if ds.base <= x < ds.base + 512]
                np.testing.assert_array_equal(visible_codes, store.codes[c])
            else:
                assert not any(ds.base <= x < ds.base + 512 for x in prompt)
        if variant == "B":
            before = [ds.control_index(int(train_indices[j])) for j in probes]
            ds.set_epoch(1)
            after = [ds.control_index(int(train_indices[j])) for j in probes]
            assert sum(a != b for a, b in zip(before, after)) > .8 * len(probes)
            ds.set_epoch(0)
            assert before == [ds.control_index(int(train_indices[j])) for j in probes]
            val_prompt = val.inference_prefix(0)
            val.set_epoch(9)
            assert val.inference_prefix(0) == val_prompt
            # Mutating the target codes must NOT change the inference prompt.
            row = int(val.indices[0]); backup = store.codes[row].copy()
            store.codes[row] = (backup + 1) % 512
            assert val.inference_prefix(0) == val_prompt
            store.codes[row] = backup
            reference = np.load(ART / "control_reference.npy")
            for j in range(min(100, len(test))):
                c = test.control_index(int(test.indices[j]))
                assert folds[c] == 4 and store.support[c] and not reference[c]
        batch = CellCollator(tok)([ds[j] for j in probes[:2]])
        assert batch["labels"].shape == (2, 67)
        assert (batch["labels"] != -100).sum(dim=1).tolist() == [66, 66]
        batches.append(batch)
        lengths[variant] = {"train_max_tokens": ds.max_length, "validation_max_tokens": val.max_length,
                            "hepg2_test_max_tokens": test.max_length}
        first = int(ds.indices[0])
        examples[variant] = {"target_row": first, "condition": store.conditions[int(cr[first])]["text"],
            "control_row": ds.control_index(first) if variant == "B" else None,
            "prompt": tok.decode(ds.inference_prefix(0), skip_special_tokens=False),
            "target": tok.decode(ds[0]["input_ids"][-66:], skip_special_tokens=False),
            "note": "Real supervised target, not a generated prediction"}
        for gpus in [2, 4]:
            for phase in ["smoke", "benchmark", "formal"]:
                cfg = runtime_config(read_json(CONFIG / f"{variant}.json"), gpus, phase)
                assert cfg["gradient_accumulation_steps"] * gpus * cfg["per_device_batch_size"] == 128
                parsed = training_arguments(cfg, ART / "cpu_argument_check", cpu_check=True)
                assert parsed.warmup_steps == .03
    # Real collator shapes with a small random language model: identical loss
    # with full logits versus cropped completion logits, and nonzero updates.
    tiny = Qwen3_5TextConfig(vocab_size=601, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, layer_types=["full_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, max_position_embeddings=256, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .25,
                         "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    torch.manual_seed(420)
    model = Qwen3_5ForCausalLM(tiny).eval()
    for batch in batches:
        ids = batch["input_ids"] % 601
        cropped_labels = batch["labels"].clone()
        cropped_labels[cropped_labels != -100] %= 601
        full_labels = torch.full_like(ids, -100)
        full_labels[:, -67:] = cropped_labels
        a = model(input_ids=ids, attention_mask=batch["attention_mask"], labels=full_labels, use_cache=False).loss
        b = model(input_ids=ids, attention_mask=batch["attention_mask"], labels=cropped_labels,
                  logits_to_keep=67, use_cache=False).loss
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        b.backward()
        assert torch.isfinite(model.get_input_embeddings().weight.grad).all()
        model.zero_grad(set_to_none=True)
    from train_replogle_llm import Progress
    class TinyTokenizer:
        def convert_tokens_to_ids(self, tokens):
            vocabulary = FSQ_TOKENS + [CELL_START, CELL_END]
            return [vocabulary.index(t) for t in tokens]
    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, index):
            return {"input_ids": ids[index % 2], "attention_mask": batch["attention_mask"][index % 2],
                    "labels": cropped_labels[index % 2], "logits_to_keep": 67}
    def collate(examples):
        return {k: (67 if k == "logits_to_keep" else torch.stack([e[k] for e in examples])) for k in examples[0]}
    cpu_out = ART / "cpu_trainer_check"
    cb = SetControlEpoch(ds)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(cpu_out), use_cpu=True,
        max_steps=4, warmup_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=2,
        learning_rate=1e-5, optim="adamw_torch", report_to="none", save_strategy="no",
        eval_strategy="no", disable_tqdm=True, remove_unused_columns=False),
        train_dataset=TinyDataset(), data_collator=collate,
        callbacks=[Progress(cpu_out, model, TinyTokenizer()), cb])
    trainer.train()
    assert read_json(cpu_out / "update_acceptance.json")["status"] == "passed" and ds.epoch == 1
    try:
        safe_output(ROOT / "outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1/formal")
        raise RuntimeError("Baseline overwrite guard failed")
    except AssertionError:
        pass
    baseline_unchanged()
    report = {"status": "passed", "device": "cpu", "gpu_checks": "not_run; required smoke gate before formal training",
        "sequence_lengths": lengths, "split_checks": "all heldout target genes remain in source train; HepG2 absent from SFT train",
        "control_sampling": "same line/gem, train-only support, excludes self, changes per epoch, validation fixed",
        "loss": "66 output tokens only; real A/B batch full-vs-cropped CE agree; finite CPU backward and optimizer updates",
        "target_leakage_check": "Changing target codes leaves B inference prefix unchanged",
        "baseline_preserved": True, "global_batch": 128, "gpus_supported": [2, 4]}
    save_json(ART / "cpu_preflight.json", report)
    save_json(ART / "case_examples.json", examples)
    # Pin all split/config/code/model inputs required by these training scripts.
    files = [p for p in ART.glob("*") if p.suffix in [".json", ".npy"] and p.name not in {"ready_inputs.json", "active_submissions.json"}]
    files += list(CONFIG.glob("*.json"))
    files += [_bootstrap.script_path(name) for name in ["prepare_replogle_control_ab.py", "replogle_control_common.py",
        "train_replogle_control_ab.py", "check_replogle_control_ab.py", "run_replogle_control_ab_worker.sh",
        "submit_replogle_qwen.sh", "submit_replogle_qwen.py", "run_replogle_qwen_worker.sh", "train_replogle_qwen.py", "replogle_qwen.py", "replogle_qwen_callbacks.py", "replogle_llm_common.py", "train_replogle_llm.py"]]
    initial = ROOT / store.sources["initial_model"]
    files += [p for p in initial.iterdir() if p.is_file()]
    files += [ROOT / store.sources["nb_cache"] / n for n in ["token_ids.npy", "condition_rows.npy", "completion.json"]]
    files += [ROOT / store.sources["text_conditions"] / "conditions.jsonl"]
    save_json(ART / "ready_inputs.json", {"status": "cpu_ready", "sha256": {
        str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
