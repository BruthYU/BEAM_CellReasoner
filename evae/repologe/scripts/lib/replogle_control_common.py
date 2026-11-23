"""Control-conditioned single-cell datasets; baseline code and caches stay read-only."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback

from replogle_llm_common import ROOT, FSQ_TOKENS, CELL_START, CELL_END, read_json, sha

NAME = "qwen35-08b-fsq-nb-hepg2-v1"
ART = ROOT / "artifacts/replogle_llm_control" / NAME
CONFIG = ROOT / "config/local/replogle_llm_control" / NAME
OUTPUT_ROOT = ROOT / "outputs/replogle_llm_control" / NAME


def check_inputs(art=ART):
    manifest = read_json(art / "ready_inputs.json")
    assert manifest["status"] == "cpu_ready"
    for name, expected in manifest["sha256"].items():
        _bootstrap.verify_input(name, expected)
    return manifest


def safe_output(path):
    path = Path(path).resolve()
    assert path.is_relative_to(OUTPUT_ROOT.resolve()) and path != OUTPUT_ROOT.resolve(), path
    return path


class CellStore:
    def __init__(self, art=ART):
        self.art = Path(art)
        self.sources = read_json(self.art / "sources.json")
        cache = ROOT / self.sources["nb_cache"]
        import json
        self.conditions = [json.loads(s) for s in (ROOT / self.sources["text_conditions"] / "conditions.jsonl").read_text().splitlines()]
        self.codes = np.load(cache / "token_ids.npy", allow_pickle=False)
        self.condition_rows = np.load(cache / "condition_rows.npy", allow_pickle=False)
        self.fold = np.load(self.art / "cell_fold.npy", allow_pickle=False)
        self.gem = np.load(self.art / "gem_group.npy", allow_pickle=False)
        self.support = np.load(self.art / "control_support.npy", allow_pickle=False)
        self.pools = {int(g): np.flatnonzero((self.gem == g) & self.support) for g in np.unique(self.gem)}
        assert all(len(v) >= 2 for v in self.pools.values())


class ControlCellDataset(Dataset):
    """An unpaired control is sampled within cell-line/gem_group, never by target expression."""
    def __init__(self, store, indices, tokenizer, variant, training=False, seed=420):
        assert variant in ["A", "B"]
        self.store, self.indices, self.tokenizer = store, np.asarray(indices, dtype=np.int64), tokenizer
        self.variant, self.training, self.seed, self.epoch = variant, training, seed, 0
        self.base = tokenizer.convert_tokens_to_ids(FSQ_TOKENS[0])
        self.start = tokenizer.convert_tokens_to_ids(CELL_START)
        self.end = tokenizer.convert_tokens_to_ids(CELL_END)
        self.eos = tokenizer.eos_token_id
        self.prefix_parts = {}
        marker = "__CONTROL_CODES_INSERTION__"
        for k in np.unique(store.condition_rows[self.indices]):
            row = store.conditions[int(k)]
            text = row["text"]
            if variant == "B":
                text += "\n\nControl Cell:\n" + marker
            text += "\n\nPredicted Cell:"
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if variant == "B":
                before, after = prompt.split(marker)
                self.prefix_parts[int(k)] = (tokenizer.encode(before, add_special_tokens=False),
                                             tokenizer.encode(after, add_special_tokens=False))
            else:
                self.prefix_parts[int(k)] = (tokenizer.encode(prompt, add_special_tokens=False), [])
        self.max_length = max(len(a) + len(b) + (66 if variant == "B" else 0) + 1 + 66
                              for a, b in self.prefix_parts.values())

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = int(epoch) if self.training else 0

    def control_index(self, row, draw=0):
        pool = self.store.pools[int(self.store.gem[row])]
        # Exclude self for non-targeting targets: never supervise identity copy.
        position = np.searchsorted(pool, row)
        is_self = position < len(pool) and pool[position] == row
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(row), int(draw), 17]))
        selected = int(rng.integers(len(pool) - int(is_self)))
        if is_self and selected >= position:
            selected += 1
        result = int(pool[selected])
        assert result != row and self.store.gem[result] == self.store.gem[row]
        if self.training:
            assert self.store.fold[row] == 0 and self.store.fold[result] == 0
        return result

    def inference_prefix(self, index, draw=0):
        """No target FSQ codes used here; row supplies condition and technical group only."""
        row = int(self.indices[index])
        before, after = self.prefix_parts[int(self.store.condition_rows[row])]
        prefix = list(before)
        if self.variant == "B":
            control_row = self.control_index(row, draw)
            prefix += [self.start] + (self.store.codes[control_row].astype(np.int64) + self.base).tolist() + [self.end]
        return prefix + after + [self.start]

    def __getitem__(self, index):
        row = int(self.indices[index])
        prefix = self.inference_prefix(index)
        target = (self.store.codes[row].astype(np.int64) + self.base).tolist() + [self.end, self.eos]
        return {"input_ids": prefix + target, "labels": [-100] * len(prefix) + target}


class SetControlEpoch(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(math.floor(state.epoch or 0))


def runtime_config(cfg, gpus, phase):
    assert gpus in [2, 4]
    cfg = dict(cfg)
    cfg["gpus"] = gpus
    denominator = cfg["per_device_batch_size"] * gpus
    assert cfg["global_batch_size"] % denominator == 0
    cfg["gradient_accumulation_steps"] = cfg["global_batch_size"] // denominator
    cfg["cpu_threads_per_process"] = max(1, cfg["total_cpu_threads"] // gpus)
    if phase == "smoke":
        cfg.update(max_steps=50, train_cell_limit=1024, eval_steps=25, save_steps=25,
                   logging_steps=5, save_only_model=True)
    elif phase == "benchmark":
        cfg.update(max_steps=200, eval_steps=200, save_steps=200, logging_steps=20, save_only_model=True)
    else:
        assert phase == "formal"
    return cfg
