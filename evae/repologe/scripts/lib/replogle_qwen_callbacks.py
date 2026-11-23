"""Shared Qwen training acceptance and throughput callbacks."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap

import json
import os
import time
import torch
from transformers import TrainerCallback
from replogle_llm_common import FSQ_TOKENS, CELL_START, CELL_END, save_json

class Progress(TrainerCallback):
    def __init__(self, output, model, tokenizer):
        self.output = output
        self.new_ids = tokenizer.convert_tokens_to_ids(FSQ_TOKENS + [CELL_START, CELL_END])
        self.initial_rows = model.get_input_embeddings().weight[self.new_ids].detach().cpu().clone()
        self.parameter_name, parameter = next((n, p) for n, p in model.named_parameters()
                                             if ".layers.0." in n and p.ndim == 2)
        self.initial_parameter = parameter.detach().flatten()[:4096].cpu().clone()
        self.checked = False

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Warmup starts with a zero-LR optimizer update. Check after updates
        # with a nonzero learning rate rather than falsely rejecting step 1.
        if not self.checked and state.global_step >= 3:
            while hasattr(model, "module"):
                model = model.module
            rows = model.get_input_embeddings().weight[self.new_ids].detach().cpu()
            row_change = float((rows - self.initial_rows).abs().max())
            parameter = dict(model.named_parameters())[self.parameter_name].detach().flatten()[:4096].cpu()
            backbone_change = float((parameter - self.initial_parameter).abs().max())
            assert row_change > 0 and backbone_change > 0, (row_change, backbone_change)
            assert torch.isfinite(rows).all()
            if state.is_world_process_zero:
                save_json(self.output / "update_acceptance.json", {
                    "status": "passed", "full_language_parameters_trainable": True,
                    "new_token_rows_updated": True, "new_token_max_change": row_change,
                    "backbone_parameter": self.parameter_name, "backbone_max_change": backbone_change,
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                    "node": os.environ.get("KUBEBRAIN_NODE_NAME"),
                })
            self.checked = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            values = {"step": state.global_step, **logs,
                      "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
            with (self.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(values, allow_nan=False) + "\n")

class Throughput(TrainerCallback):
    def __init__(self, output, batch):
        self.output, self.batch = output, batch
        self.start = None

    def on_step_end(self, args, state, control, **kwargs):
        # Exclude optimizer warmup/JIT startup; measure identical optimizer steps.
        if state.global_step == 50:
            torch.cuda.synchronize()
            self.start = time.monotonic()
        if self.start is not None and state.global_step == 200:
            torch.cuda.synchronize()
            elapsed = time.monotonic() - self.start
            if state.is_world_process_zero:
                save_json(self.output / "throughput.json", {"measured_steps": 150,
                    "elapsed_seconds": elapsed, "cells_per_second": 150 * self.batch / elapsed,
                    "optimizer_seconds_per_step": elapsed / 150,
                    "gpus": int(os.environ["WORLD_SIZE"]), "gpu": torch.cuda.get_device_name(),
                    "global_batch": self.batch})
