"""C: N unpaired control cells condition one output cell; A/B remain immutable."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


from pathlib import Path

import numpy as np

from replogle_llm_common import ROOT, read_json, sha
from replogle_control_common import (ART as SPLIT_ART, CellStore, ControlCellDataset,
    runtime_config as ab_runtime_config, check_inputs as check_ab_inputs)

NAME = "qwen35-08b-fsq-nb-hepg2-c-v1"
ART = ROOT / "artifacts/replogle_llm_control" / NAME
CONFIG = ROOT / "config/local/replogle_llm_control" / NAME / "C.json"
OUTPUT_ROOT = ROOT / "outputs/replogle_llm_control" / NAME


def positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("Expected a positive integer")
    return value


def runtime_config(cfg, gpus, phase, control_cells=None, per_device_batch_size=None):
    cfg = dict(cfg)
    cfg["control_cells"] = positive_int(control_cells if control_cells is not None else cfg["control_cells"])
    if per_device_batch_size is not None:
        cfg["per_device_batch_size"] = positive_int(per_device_batch_size)
        cfg["eval_batch_size"] = cfg["per_device_batch_size"]
    # Allow all N complete cells: never silently truncate the prompt.
    cfg["max_sequence_length"] = 256 + 66 * cfg["control_cells"]
    sources = read_json(SPLIT_ART / "sources.json")
    context = read_json(ROOT / sources["initial_model"] / "config.json")["max_position_embeddings"]
    if cfg["max_sequence_length"] > context:
        raise ValueError(f"N={cfg['control_cells']} exceeds the model context budget {context}")
    return ab_runtime_config(cfg, gpus, phase)


def safe_output(path):
    path = Path(path).resolve()
    assert path.is_relative_to(OUTPUT_ROOT.resolve()) and path != OUTPUT_ROOT.resolve(), path
    return path


def check_inputs():
    check_ab_inputs()
    manifest = read_json(ART / "ready_inputs.json")
    assert manifest["status"] == "cpu_ready"
    for name, expected in manifest["sha256"].items():
        _bootstrap.verify_input(name, expected)


class ControlSetDataset(ControlCellDataset):
    """Ordered random serialization of a control set, with completion-only loss."""

    def __init__(self, store, indices, tokenizer, control_cells=8, training=False, seed=420):
        self.control_cells = positive_int(control_cells)
        super().__init__(store, indices, tokenizer, "B", training=training, seed=seed)
        self.variant = "C"
        marker = "__CONTROL_CODES_INSERTION__"
        for k in self.prefix_parts:
            text = store.conditions[k]["text"]
            text += f"\n\nNumber of Control Cells: {self.control_cells}\nControl Cells:\n{marker}"
            text += "\n\nPredicted Cell:"
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            before, after = prompt.split(marker)
            self.prefix_parts[k] = (tokenizer.encode(before, add_special_tokens=False),
                                    tokenizer.encode(after, add_special_tokens=False))
        self.max_length = max(len(a) + len(b) + 66 * self.control_cells + 67
                              for a, b in self.prefix_parts.values())

    def control_indices(self, row, draw=0):
        pool = self.store.pools[int(self.store.gem[row])]
        pool = pool[pool != row]
        assert len(pool), "No eligible control after excluding target cell"
        if self.control_cells == 1:
            # Same sampled reference as B for an N=1 sanity comparison.
            return np.array([super().control_index(row, draw)], dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(row), int(draw), 17]))
        if len(pool) >= self.control_cells:
            selected = rng.choice(pool, self.control_cells, replace=False)
        else:
            # Include every available reference, then fill the shortfall with replacement.
            selected = np.concatenate([pool, rng.choice(pool, self.control_cells - len(pool), replace=True)])
            rng.shuffle(selected)
        assert (selected != row).all() and (self.store.gem[selected] == self.store.gem[row]).all()
        if self.training:
            assert self.store.fold[row] == 0 and (self.store.fold[selected] == 0).all()
        return selected

    def inference_prefix(self, index, draw=0):
        row = int(self.indices[index])
        before, after = self.prefix_parts[int(self.store.condition_rows[row])]
        prefix = list(before)
        for control_row in self.control_indices(row, draw):
            prefix += [self.start] + (self.store.codes[control_row].astype(np.int64) + self.base).tolist() + [self.end]
        return prefix + after + [self.start]
