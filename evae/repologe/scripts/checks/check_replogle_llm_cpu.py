"""Check dedicated vocabulary, all prompts, and completion-loss alignment on CPU."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import json

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, Trainer, TrainingArguments

from replogle_llm_common import (
    DEFAULT_ART, ROOT, FSQ_TOKENS, CELL_START, CELL_END,
    prepare_tokenizer, condition_prefixes, read_json, save_json, training_arguments,
    CellDataset, CellCollator,
)


def main():
    torch.set_num_threads(2)
    art = DEFAULT_ART
    tokenizer = prepare_tokenizer(art / "initial_text_model")
    source = read_json(art / "sources.json")
    rows = [json.loads(line) for line in (ROOT / source["text_conditions"] / "conditions.jsonl").read_text().splitlines()]
    prefixes = condition_prefixes(tokenizer, rows)
    for row, prefix in zip(rows, prefixes):
        assert row["text"] in tokenizer.decode(prefix, skip_special_tokens=False)
    lengths = [len(p) + 66 for p in prefixes]
    assert max(lengths) <= 256
    assert len(set(tokenizer.convert_tokens_to_ids(FSQ_TOKENS + [CELL_START, CELL_END]))) == 514
    tiny = Qwen3_5TextConfig(
        vocab_size=601, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        layer_types=["full_attention"], num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, max_position_embeddings=128, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 2],
                         "mrope_interleaved": True},
    )
    torch.manual_seed(420)
    model = Qwen3_5ForCausalLM(tiny).eval()
    ids = torch.randint(0, 601, (2, 16))
    mask = torch.ones_like(ids)
    mask[1, :2] = 0
    labels = ids.clone()
    labels[:, :-4] = -100
    full = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    cropped = model(input_ids=ids, attention_mask=mask, labels=labels[:, -5:], logits_to_keep=5, use_cache=False).loss
    torch.testing.assert_close(full, cropped, rtol=1e-6, atol=1e-6)
    cropped.backward()
    assert torch.isfinite(model.get_input_embeddings().weight.grad).all()
    tokenizer.save_pretrained(art / "tokenizer")
    config_paths = sorted((ROOT / "config/local/replogle_llm/qwen35-08b-ar-fsq-nb-v1").glob("*.json"))
    for path in config_paths:
        cfg = read_json(path)
        parsed = training_arguments(cfg, art / "cpu_argument_check", cpu_check=True)
        assert parsed.warmup_steps == 0.03 and parsed.dataloader_num_workers == 0
    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"input_ids": ids[index % 2], "attention_mask": mask[index % 2],
                    "labels": labels[index % 2, -5:], "logits_to_keep": 5}

    def collate(examples):
        return {name: (5 if name == "logits_to_keep" else torch.stack([e[name] for e in examples]))
                for name in examples[0]}

    model.zero_grad(set_to_none=True)
    from train_replogle_llm import Progress
    class TinyTokenizer:
        def convert_tokens_to_ids(self, tokens):
            vocabulary = FSQ_TOKENS + [CELL_START, CELL_END]
            return [vocabulary.index(token) for token in tokens]

    cpu_output = art / "cpu_trainer_check"
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(cpu_output), use_cpu=True, max_steps=4, warmup_steps=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        learning_rate=1e-5, optim="adamw_torch", report_to="none",
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        remove_unused_columns=False,
    ), train_dataset=TinyDataset(), data_collator=collate,
       callbacks=[Progress(cpu_output, model, TinyTokenizer())])
    trainer_result = trainer.train()
    assert torch.isfinite(torch.tensor(trainer_result.training_loss))
    assert read_json(cpu_output / "update_acceptance.json")["status"] == "passed"
    if (art / "cache/completion.json").exists():
        import numpy as np
        fold = np.load(art / "cell_fold.npy")
        selected = [int(np.flatnonzero(fold == i)[0]) for i in [0, 1, 2]]
        ds = CellDataset(art, selected, tokenizer)
        batch = CellCollator(tokenizer)([ds[i] for i in range(3)])
        assert batch["labels"].shape == (3, 67) and batch["logits_to_keep"] == 67
        assert bool((batch["labels"] != -100).sum(dim=1).eq(66).all())
    report = {"status": "passed", "device": "cpu", "conditions": len(rows),
        "dedicated_tokens": 514, "fsq_tokens": 512, "fsq_base_id": tokenizer.convert_tokens_to_ids(FSQ_TOKENS[0]),
        "cell_start_id": tokenizer.convert_tokens_to_ids(CELL_START), "cell_end_id": tokenizer.convert_tokens_to_ids(CELL_END),
        "vocab_size": len(tokenizer), "min_sequence_length": min(lengths), "max_sequence_length": max(lengths),
        "sequence_limit": 256, "thinking": False,
        "condition_text_roundtrip": "Exact approved four-line text present in every decoded prompt",
        "training_config_checks": [p.name for p in config_paths],
        "tiny_cpu_trainer_update": "passed; 4 optimizer updates, accumulation 2, zero-LR warmup and update acceptance",
        "real_cached_cell_collator": "passed" if (art / "cache/completion.json").exists() else "cache not ready",
        "loss_alignment": "Last 67 logits equal full-sequence completion-only CE; backward finite",
        "test_uses_random_tiny_model": True, "real_gpu_model_validation": "Required in smoke training"}
    save_json(art / "cpu_preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
