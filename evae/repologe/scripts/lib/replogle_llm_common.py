"""Shared single-cell SFT data and ancestral sampling; FSQ stays unchanged."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ART = ROOT / "artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
FSQ_TOKENS = [f"<fsq_{i}>" for i in range(512)]
CELL_START, CELL_END = "<cell_start>", "<cell_end>"


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def install_fast_delta():
    """Use the installed FLA kernels without downloading hub kernels."""
    import triton
    from packaging.version import Version
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
        assert Version(triton.__version__) >= Version("3.7.1"), "Hopper GDN backward requires corrected Triton"
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    qwen.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    qwen.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule


def prepare_tokenizer(model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    tokenizer.add_special_tokens({"additional_special_tokens": FSQ_TOKENS + [CELL_START, CELL_END]})
    ids = tokenizer.convert_tokens_to_ids(FSQ_TOKENS)
    assert len(set(ids)) == 512 and ids == list(range(ids[0], ids[0] + 512))
    for token in FSQ_TOKENS + [CELL_START, CELL_END]:
        assert tokenizer.encode(token, add_special_tokens=False) == [tokenizer.convert_tokens_to_ids(token)]
    return tokenizer


def condition_prefixes(tokenizer, conditions):
    start = tokenizer.convert_tokens_to_ids(CELL_START)
    result = []
    for row in conditions:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["text"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        assert row["text"] in rendered
        prefix = tokenizer.encode(rendered, add_special_tokens=False)
        assert prefix and all(isinstance(token, int) for token in prefix)
        result.append(prefix + [start])
    return result


class CellDataset(Dataset):
    def __init__(self, art, indices, tokenizer):
        self.art = Path(art)
        self.indices = np.asarray(indices, dtype=np.int64)
        # The entire uint16 cache is only 118 MiB. Reading it normally also works
        # on the interactive host's shared mount, which rejects shared mmap.
        self.codes = np.load(self.art / "cache/token_ids.npy", allow_pickle=False)
        self.condition_rows = np.load(self.art / "cache/condition_rows.npy", allow_pickle=False)
        self.conditions = [json.loads(line) for line in
                           (ROOT / read_json(self.art / "sources.json")["text_conditions"] / "conditions.jsonl").read_text().splitlines()]
        self.prefixes = condition_prefixes(tokenizer, self.conditions)
        self.code_base = tokenizer.convert_tokens_to_ids(FSQ_TOKENS[0])
        self.cell_end = tokenizer.convert_tokens_to_ids(CELL_END)
        self.eos = tokenizer.eos_token_id
        self.max_length = max(map(len, self.prefixes)) + 66
        assert self.max_length <= 256, self.max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        prefix = self.prefixes[int(self.condition_rows[row])]
        target = (self.codes[row].astype(np.int64) + self.code_base).tolist() + [self.cell_end, self.eos]
        return {"input_ids": prefix + target, "labels": [-100] * len(prefix) + target}


class CellCollator:
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id

    def __call__(self, examples):
        length = (max(len(x["input_ids"]) for x in examples) + 7) // 8 * 8
        ids, masks, labels = [], [], []
        for x in examples:
            padding = length - len(x["input_ids"])
            ids.append([self.pad] * padding + x["input_ids"])
            masks.append([0] * padding + [1] * len(x["input_ids"]))
            labels.append([-100] * padding + x["labels"])
        labels = torch.tensor(labels, dtype=torch.long)
        assert bool((labels != -100).sum(dim=1).eq(66).all())
        # All completions are 66 tokens. Keep their preceding prediction position,
        # avoiding unused prompt logits while retaining exactly the same CE loss.
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
                "labels": labels[:, -67:], "logits_to_keep": 67}


@torch.inference_mode()
def sample_codes(model, tokenizer, prefixes, temperature=1.0):
    from transformers import LogitsProcessor, LogitsProcessorList

    base = tokenizer.convert_tokens_to_ids(FSQ_TOKENS[0])

    class CodesOnly(LogitsProcessor):
        def __call__(self, input_ids, scores):
            allowed = scores[:, base:base + 512].clone()
            scores.fill_(-torch.inf)
            scores[:, base:base + 512] = allowed
            return scores

    maximum = max(map(len, prefixes))
    input_ids = torch.tensor([[tokenizer.pad_token_id] * (maximum - len(p)) + p for p in prefixes], device=model.device)
    mask = torch.tensor([[0] * (maximum - len(p)) + [1] * len(p) for p in prefixes], device=model.device)
    generated = model.generate(
        input_ids=input_ids, attention_mask=mask, max_new_tokens=64, min_new_tokens=64,
        do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
        logits_processor=LogitsProcessorList([CodesOnly()]), use_cache=True,
        disable_compile=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=None,
    )
    codes = generated[:, maximum:] - base
    assert codes.shape == (len(prefixes), 64) and codes.min() >= 0 and codes.max() < 512
    return codes


def validation_indices(art, cap=20):
    art = Path(art)
    fold = np.load(art / "cell_fold.npy")
    conditions = np.load(art / "cache/condition_rows.npy", allow_pickle=False)
    rng = np.random.default_rng(420)
    chosen = []
    for condition in read_json(art / "split.json")["validation_condition_rows"]:
        candidates = np.flatnonzero((fold == 1) & (conditions == condition))
        chosen.extend(rng.choice(candidates, min(cap, len(candidates)), replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def training_arguments(cfg, output, cpu_check=False):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=str(output), seed=cfg["seed"], data_seed=cfg["seed"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=cfg.get("eval_batch_size", 8),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.01),
        # Transformers 5.23 accepts a ratio through warmup_steps; warmup_ratio was removed.
        warmup_steps=cfg.get("warmup_ratio", 0.03), lr_scheduler_type="cosine",
        max_steps=cfg.get("max_steps", -1), num_train_epochs=cfg.get("epochs", 3),
        bf16=not cpu_check, use_cpu=cpu_check, bf16_full_eval=False, fp16=False,
        optim="adamw_torch_fused", max_grad_norm=1.0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, dataloader_num_workers=0, dataloader_pin_memory=True,
        remove_unused_columns=False, prediction_loss_only=True,
        eval_strategy="steps", eval_steps=cfg["eval_steps"],
        save_strategy="steps", save_steps=cfg["save_steps"], save_total_limit=2,
        save_only_model=cfg.get("save_only_model", False), load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=cfg.get("logging_steps", 20), report_to="none",
    )
