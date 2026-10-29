"""Small distributed/scheduler helpers used by VQVAE.fit_ddp_advanced."""

import math
import random

import numpy as np
import torch
import torch.distributed as dist

from lib_vqvae import utils


def distributed_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def warmup_cosine_factor(step, total_steps, warmup_steps, min_lr_ratio=0.1):
    """LR for zero-based optimizer step; warmup never starts with a zero update."""
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError("require total_steps >= 1 and 0 <= warmup_steps < total_steps")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    # The last optimizer update uses min_lr_ratio. With no warmup, the first
    # update uses the full LR, rather than starting one cosine step late.
    if warmup_steps:
        position = (step - warmup_steps + 1) / (total_steps - warmup_steps)
    else:
        position = step / max(total_steps - 1, 1)
    position = min(max(position, 0.0), 1.0)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * position))


def make_warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio=0.03, min_lr_ratio=0.1):
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = min(math.ceil(total_steps * warmup_ratio), total_steps - 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_factor(step, total_steps, warmup_steps, min_lr_ratio),
    )
    return scheduler, warmup_steps


def gather_records(record):
    _, world_size = distributed_info()
    if world_size == 1:
        return [record]
    records = [None] * world_size
    dist.all_gather_object(records, record)
    return records


@torch.no_grad()
def advanced_validation(model, loader, mode, device):
    """As in fit: evaluate one batch, now weighted correctly across all ranks."""
    _, world_size = distributed_info()
    model.eval()
    batch = next(iter(loader), None)
    present = torch.tensor(int(batch is not None), device=device)
    if world_size > 1:
        dist.all_reduce(present, op=dist.ReduceOp.MIN)
    if not present.item():
        raise ValueError("every rank needs a non-empty validation loader")
    x = batch.float().to(device, non_blocking=True)
    outputs = model(x, mode, device)
    loss = torch.nn.functional.mse_loss(outputs[3], x) + outputs[0]
    values = torch.stack((loss.double() * len(x), loss.new_tensor(len(x)).double()))
    if world_size > 1:
        dist.all_reduce(values)
    result = (values[0] / values[1]).item()
    if not math.isfinite(result):
        raise FloatingPointError("non-finite distributed validation loss")
    model.train()
    return result


@torch.no_grad()
def parameter_sync_error(model):
    """Read-only assertion of DDP parameter agreement, performed once per epoch."""
    _, world_size = distributed_info()
    if world_size == 1:
        return 0.0
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    reference = flat.clone()
    dist.broadcast(reference, src=0)
    difference = (flat - reference).abs().max()
    dist.all_reduce(difference, op=dist.ReduceOp.MAX)
    return difference.item()


def save_advanced_checkpoint(model, optimizer, scheduler, results, args, epoch,
                             global_step, settings, device):
    rank, world_size = distributed_info()
    local_rng = {
        "torch": torch.get_rng_state(), "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
    }
    rng_states = gather_records(local_rng)
    failure = [None]
    if rank == 0:
        try:
            utils.save_model_and_results(
                model, None, None, results, vars(args), epoch, args,
                training_state={
                    "epoch": epoch, "global_step": global_step,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "advanced_settings": settings, "world_size": world_size,
                    "rng_state_by_rank": rng_states,
                },
            )
        except Exception as exc:
            failure[0] = repr(exc)
    if world_size > 1:
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise OSError(f"checkpoint save failed on rank 0: {failure[0]}")


def restore_advanced_checkpoint(path, model, optimizer, scheduler, settings, device):
    rank, world_size = distributed_info()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["world_size"] != world_size:
        raise ValueError("exact resume requires the checkpoint's original WORLD_SIZE")
    previous = checkpoint["advanced_settings"]
    for key in ("global_batch_size", "steps_per_epoch", "epochs", "learning_rate",
                "weight_decay", "warmup_steps", "min_lr_ratio", "optimizer", "scheduler"):
        if settings[key] != previous[key]:
            raise ValueError(f"resume configuration mismatch: {key}")
    for key in ("momentum", "dampening", "nesterov", "betas", "eps", "plateau_factor",
                "plateau_patience", "eval_every_steps", "gradient_clip_norm", "warmup_ratio"):
        if settings.get(key) != previous.get(key):
            raise ValueError(f"resume configuration mismatch: {key}")
    # Older AdamW checkpoints predate this explicit field, but used total_steps.
    if settings.get("scheduler_total_steps", settings["total_optimizer_steps"]) != previous.get("scheduler_total_steps", previous["total_optimizer_steps"]):
        raise ValueError("resume configuration mismatch: scheduler_total_steps")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    state = checkpoint["rng_state_by_rank"][rank]
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)
    return checkpoint["epoch"], checkpoint["global_step"], checkpoint["results"]
