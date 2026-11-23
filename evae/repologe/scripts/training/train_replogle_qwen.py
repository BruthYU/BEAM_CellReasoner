"""One full-SFT implementation for all Qwen NB-FSQ prompt/split presets."""
# 内部训练内核。日常调用：
# bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-c --control-cells-nums 8 --dry-run
# GPU任务内直接调用（卡数须与请求JSON一致）：
# torchrun --standalone --nproc-per-node=4 scripts/training/train_replogle_qwen.py --config REQUEST_JSON --phase formal --output RUN_DIR/formal

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap

import argparse
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM, Trainer, set_seed
from replogle_llm_common import (ROOT, DEFAULT_ART, FSQ_TOKENS, CELL_START, CELL_END,
    CellCollator, prepare_tokenizer, install_fast_delta, read_json, save_json, sample_codes)
from replogle_control_common import SetControlEpoch
from replogle_control_baseline import make_training_arguments
from replogle_qwen_callbacks import Progress, Throughput
from replogle_qwen import training_data, make_dataset, verify_inputs, safe_output, phase_config


def train(cfg, output, *, verify=None, resume=None):
    output = Path(output).resolve()
    verify = verify or (lambda: verify_inputs(cfg))
    cfg = dict(cfg)
    cfg.setdefault("gradient_checkpointing", True)
    cfg.setdefault("phase", "formal")
    cfg.setdefault("global_batch_size", cfg["gpus"] * cfg["per_device_batch_size"] * cfg["gradient_accumulation_steps"])
    assert int(os.environ["WORLD_SIZE"]) == cfg["gpus"]
    assert os.environ.get("KUBEBRAIN_QUOTA_GROUP") == cfg["charged_group"]
    if cfg.get("node"):
        assert os.environ.get("KUBEBRAIN_NODE_NAME") == cfg.get("node")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(cfg["cpu_threads_per_process"])
    if rank == 0:
        verify()
    set_seed(cfg["seed"])
    install_fast_delta()
    ta = make_training_arguments(cfg, output)
    _ = ta.device
    if rank == 0:
        if resume:
            checkpoint = Path(resume).resolve()
            assert checkpoint.is_relative_to(output) and (checkpoint / "trainer_state.json").exists()
            assert not (output / "completion.json").exists(), "Completed runs are immutable"
            previous = read_json(output / "run_config.json")
            for key in ("preset", "variant", "control_cells", "global_batch_size", "seed"):
                if key in previous:
                    assert previous[key] == cfg[key], f"Resume changes {key}"
        else:
            assert not output.exists() or not any(output.iterdir()), f"Refusing to overwrite {output}"
            output.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    store, fold, sources, monitor = training_data(cfg)
    initial = ROOT / sources.get("initial_model", str(DEFAULT_ART.relative_to(ROOT) / "initial_text_model"))
    assert not sources.get("uses_existing_sft_weights", False)
    assert initial.name == "initial_text_model"
    tokenizer = prepare_tokenizer(initial)
    model = Qwen3_5ForCausalLM.from_pretrained(initial, local_files_only=True,
        dtype=torch.float32, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.tie_weights()
    ids = tokenizer.convert_tokens_to_ids(FSQ_TOKENS + [CELL_START, CELL_END])
    with torch.no_grad():
        embedding = model.get_input_embeddings().weight
        embedding[ids] = torch.randn(len(ids), embedding.shape[1], dtype=embedding.dtype) * model.config.initializer_range
    assert all(p.requires_grad for p in model.parameters())
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    model.config.use_cache = False
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    indices = np.flatnonzero(fold == 0)
    if cfg.get("train_cell_limit"):
        indices = np.random.default_rng(cfg["seed"]).choice(indices, cfg["train_cell_limit"], replace=False)
    if cfg["phase"] == "smoke":
        monitor = monitor[:256]
    train = make_dataset(cfg, store, indices, tokenizer, training=True)
    validation = make_dataset(cfg, store, monitor, tokenizer)
    assert max(train.max_length, validation.max_length) <= cfg["max_sequence_length"]
    assert (fold[indices] == 0).all() and (fold[monitor] == 1).all()
    callbacks = [Progress(output, model, tokenizer), Throughput(output, cfg["global_batch_size"])]
    if hasattr(train, "set_epoch"):
        callbacks.append(SetControlEpoch(train))
    trainer = Trainer(model=model, args=ta, train_dataset=train, eval_dataset=validation,
        data_collator=CellCollator(tokenizer), processing_class=tokenizer,
        callbacks=callbacks)
    if trainer.is_world_process_zero():
        save_json(output / "run_config.json", {**cfg, "train_cells": len(train),
            "validation_monitor_cells": len(validation), "sequence_max_length": train.max_length,
            "initial_model": str(initial), "initial_model_sha256": sources.get("initial_model_sha256"),
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "test_used_for_selection": False, "automatic_population_evaluation": False})
    start = time.monotonic()
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    metrics = trainer.evaluate() if cfg["phase"] != "benchmark" else {}
    if cfg["phase"] != "benchmark":
        trainer.save_model(str(output / "final"))
        trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    # Measure every rank, not just rank 0; also retain CUDA allocator reservation.
    memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                          device=torch.cuda.current_device(), dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(memory, op=torch.distributed.ReduceOp.MAX)
    if trainer.is_world_process_zero():
        assert read_json(output / "update_acceptance.json")["status"] == "passed"
        assert math.isfinite(result.training_loss)
        if metrics:
            assert math.isfinite(metrics["eval_loss"])
        if cfg["phase"] == "smoke":
            model = trainer.accelerator.unwrap_model(trainer.model)
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                codes = sample_codes(model, tokenizer, [validation.inference_prefix(i) for i in range(4)])
            save_json(output / "generation_acceptance.json", {"status": "passed",
                "shape": list(codes.shape), "range": [int(codes.min()), int(codes.max())],
                "control_cells": cfg["control_cells"], "target_codes_in_prompt": False})
        save_json(output / "completion.json", {"status": "complete", "phase": cfg["phase"],
            "steps": trainer.state.global_step, "best_checkpoint": trainer.state.best_model_checkpoint,
            "train_metrics": result.metrics, "validation_metrics": metrics,
            "elapsed_seconds": time.monotonic() - start, "gpus": cfg["gpus"],
            "gpu": torch.cuda.get_device_name(), "peak_allocated_gib": memory[0].item() / 1024**3,
            "peak_reserved_gib": memory[1].item() / 1024**3,
            "node": os.environ.get("KUBEBRAIN_NODE_NAME"), "charged_group": cfg["charged_group"],
            "test_evaluation_performed": False, "final_model": str(output / "final")})
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.end_training()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--phase", choices=["smoke", "benchmark", "formal"], default="formal")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    requested = read_json(args.config)
    requested.setdefault("preset", "original-control1")
    cfg = phase_config(requested, args.phase)
    train(cfg, safe_output(args.output, cfg), resume=args.resume)


if __name__ == "__main__":
    main()
