import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

import lib_vqvae.utils as utils
from lib_vqvae.model import Encoder, Decoder
from lib_vqvae.dataset import create_dataset_loader
from lib_vqvae.quantizer import VectorQuantizer

try:
    import wandb
except ImportError:  # Optional for the legacy ``fit`` helper.
    wandb = None


class VQVAE(nn.Module):
    def __init__(self, data_dim, n=4, codebook_dim=16, n_codebooks=512, encoder_hidden_dim=[1600,1024,800], decoder_hidden_dim=[800,1024,1600], beta=1, gamma=1, dropout_rate=0.2):
        super(VQVAE, self).__init__()
        self.data_dim = data_dim
        self.n = n
        self.codebook_dim = codebook_dim
        self.n_codebooks = n_codebooks
        self.dropout_rate=dropout_rate

        self.encoder = Encoder(self.data_dim, self.n, self.codebook_dim, encoder_hidden_dim, self.dropout_rate)
        self.vector_quantization = VectorQuantizer(self.n, self.n_codebooks, self.codebook_dim, beta, gamma)
        self.decoder = Decoder(self.data_dim, self.n, self.codebook_dim, decoder_hidden_dim, self.dropout_rate)


    def forward(self, x, mode, device, verbose=False):
        z_e = self.encoder(x)
        embedding_loss, loss1, loss2, z_q, z_output, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(
            z_e, mode, device, training=self.training
        )
        x_hat = self.decoder(z_output)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('vq data shape:', z_q.shape)
            print('before decoded data shape', z_output.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, loss1, loss2, x_hat, z_q, z_output, perplexity, min_encodings, min_encoding_indices


    def eval_step(self, eval_loader, mode, device):
        # - eval step
        # self.to(device)
        self.eval()
        step_total_loss = 0
        batches = 0

        with torch.no_grad():
            for i, x in enumerate(eval_loader):
                x = x.float().to(device)

                embedding_loss, loss1, loss2, x_hat, z_q, z_output, perplexity, min_encodings, min_encoding_indices=self(x, mode, device)

                recon_loss = F.mse_loss(x_hat, x)
                loss = recon_loss + embedding_loss

                step_total_loss += loss.item()
                batches += 1
                break
        if not batches:
            raise ValueError("eval_loader must contain at least one batch")
        return step_total_loss


    def fit(self, args, dataloader, eval_adata, eval_loader, mode, device, max_iter=300, lr=3e-4, weight_decay=5e-4, log_val=10, save=True):
        """Train on one device using the released SGD/loss/validation protocol.

        Progress spans all epochs, so tqdm's remaining time is the whole-run
        ETA. progress.json and metrics.jsonl also work without a terminal.
        """
        print("device:", device)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

        use_wandb = wandb is not None and getattr(wandb, "run", None) is not None
        if use_wandb:
            wandb.watch(self, log="all")

        results = {
        'n_updates': [],
        'recon_loss': [],
        'embedding_loss': [],
        'train_loss': [],
        'perplexities': [],
        'eval_loss': [],
                    }

        totallen = len(dataloader)
        if totallen < 1 or max_iter < 1:
            raise ValueError("fit requires non-empty training data and max_iter >= 1")
        save_interval = int(getattr(args, "log_val", log_val))
        if save_interval < 1:
            raise ValueError("log_val must be >= 1")
        output = Path(args.outdir) if getattr(args, "outdir", None) else None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        last_report = started
        total_steps = max_iter * totallen
        completed = 0
        cells_seen = 0
        state = {}

        def report(status, epoch, step, loss_value=None):
            elapsed = time.monotonic() - started
            seconds_per_step = elapsed / completed if completed else None
            state.update({
                "event": "fit_progress", "status": status,
                "epoch": epoch, "epochs": max_iter,
                "epoch_step": step, "steps_per_epoch": totallen,
                "completed_steps": completed, "total_steps": total_steps,
                "cells_seen": cells_seen, "elapsed_seconds": elapsed,
                "cells_per_second": cells_seen / max(elapsed, 1e-9),
                "eta_seconds": (total_steps - completed) * seconds_per_step
                if seconds_per_step is not None else None,
                "epoch_eta_seconds": (totallen - step) * seconds_per_step
                if seconds_per_step is not None else None,
                "loss": loss_value,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "device": str(device), "pid": os.getpid(),
            })
            if output is not None:
                temporary = output / "progress.json.tmp"
                temporary.write_text(json.dumps(state, indent=2) + "\n")
                os.replace(temporary, output / "progress.json")
            tqdm.write(json.dumps(state), file=sys.stdout)

        report("running", 1, 0)
        try:
            with tqdm(total=total_steps, desc="VQVAE.fit", unit="batch",
                      dynamic_ncols=True, mininterval=2, file=sys.stdout) as progress:
                for epoch in range(max_iter):
                    self.train()
                    sampler = getattr(dataloader, "batch_sampler", None)
                    if hasattr(sampler, "set_epoch"):
                        sampler.set_epoch(epoch)
                    epoch_reconloss, epoch_embeddloss, epoch_perplexities = 0, 0, 0
                    validation_count = len(results["eval_loss"])
                    epoch_started = time.monotonic()
                    epoch_cells = 0
                    for i, x in enumerate(dataloader):
                        x = x.float().to(device)
                        optimizer.zero_grad()
                        embedding_loss, loss1, loss2, x_hat, z_q, z_output, perplexity, min_encodings, min_encoding_indices = self(x, mode, device)
                        recon_loss = F.mse_loss(x_hat, x)
                        loss = recon_loss + embedding_loss
                        if not torch.isfinite(loss).item():
                            raise FloatingPointError(f"non-finite loss at epoch={epoch+1}, step={i+1}")
                        loss.backward()
                        optimizer.step()
                        loss_value = loss.item()
                        epoch_reconloss += recon_loss.item()
                        epoch_embeddloss += embedding_loss.item()
                        epoch_perplexities += perplexity.item()
                        cells_seen += len(x)
                        epoch_cells += len(x)
                        completed += 1
                        if (i+1) % 100 == 0:
                            eval_loss = self.eval_step(eval_loader, mode, device)
                            if not np.isfinite(eval_loss):
                                raise FloatingPointError("non-finite validation loss")
                            results["eval_loss"].append(eval_loss)
                            if use_wandb:
                                wandb.log({"eval_loss": eval_loss})
                            self.train()
                        progress.set_description(f"VQVAE.fit epoch {epoch+1}/{max_iter}", refresh=False)
                        progress.set_postfix(loss=f"{loss_value:.4f}", epoch_step=f"{i+1}/{totallen}", refresh=False)
                        progress.update(1)
                        if time.monotonic() - last_report >= 10 or completed == 1:
                            report("running", epoch+1, i+1, loss_value)
                            last_report = time.monotonic()

                    # Short smoke datasets still require validation for the scheduler.
                    if len(results["eval_loss"]) == validation_count:
                        eval_loss = self.eval_step(eval_loader, mode, device)
                        if not np.isfinite(eval_loss):
                            raise FloatingPointError("non-finite validation loss")
                        results["eval_loss"].append(eval_loss)
                        self.train()
                    scheduler.step(results["eval_loss"][-1])
                    results["recon_loss"].append(epoch_reconloss/totallen)
                    results["embedding_loss"].append(epoch_embeddloss/totallen)
                    results["perplexities"].append(epoch_perplexities/totallen)
                    results["train_loss"].append((epoch_reconloss+epoch_embeddloss)/totallen)
                    results["n_updates"].append(epoch+1)
                    record = {
                        "event": "fit_epoch", "epoch": epoch+1,
                        "steps": totallen, "cells": epoch_cells,
                        "seconds": time.monotonic() - epoch_started,
                        "reconstruction_loss": results["recon_loss"][-1],
                        "embedding_loss": results["embedding_loss"][-1],
                        "total_loss": results["train_loss"][-1],
                        "perplexity": results["perplexities"][-1],
                        "validation_loss": results["eval_loss"][-1],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    tqdm.write(json.dumps(record), file=sys.stdout)
                    if output is not None:
                        with (output / "metrics.jsonl").open("a") as handle:
                            handle.write(json.dumps(record) + "\n")
                    if use_wandb:
                        wandb.log(record)
                    if save and ((epoch+1) % save_interval == 0 or epoch+1 == max_iter):
                        utils.save_model_and_results(
                            self, eval_adata, eval_loader, results, vars(args), epoch+1, args,
                            training_state={
                                "epoch": epoch+1,
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "rng_state": {"torch": torch.get_rng_state(),
                                    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                                    "numpy": np.random.get_state()},
                            },
                        )
                    report("running", epoch+1, totallen, loss_value)
            report("complete", max_iter, totallen, loss_value)
        except BaseException:
            report("failed", state.get("epoch", 1), state.get("epoch_step", 0), state.get("loss"))
            raise
        return results


    def fit_ddp_advanced(
        self, args, dataloader, eval_adata, eval_loader, mode, device,
        max_iter=50, lr=3e-4, weight_decay=5e-4, log_val=5, save=True,
        global_batch_size=512, warmup_ratio=0.03, min_lr_ratio=0.1,
        eval_every_steps=100, gradient_clip_norm=None, resume=None,
        optimizer_name="adamw", scheduler_name="warmup_cosine", momentum=0.9,
        plateau_factor=0.5, plateau_patience=50, scheduler_epochs=None,
    ):
        """Fit the same reconstruction + VQ objective on one/N GPUs.

        AdamW + warmup/cosine remains the unchanged default. SGD and an
        epoch-level ReduceLROnPlateau are optional for reproduction controls.
        The entry point starts one process per GPU and prepares disjoint loaders.
        DDP averages gradients: each rank uses global_batch_size/WORLD_SIZE and
        its unscaled local mean loss. Warmup/cosine advances once per global
        optimizer update, not once per GPU or validation call. Only rank 0
        writes progress/checkpoints. Ordinary vqvae has no rank-local online
        codebook mutations; cvqvae's online reinitialisation is not DDP-safe.
        """
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        from lib_vqvae.advanced_training import (
            advanced_validation, distributed_info, gather_records,
            make_warmup_cosine_scheduler, parameter_sync_error,
            restore_advanced_checkpoint, save_advanced_checkpoint,
        )

        rank, world_size = distributed_info()
        device = torch.device(device)
        optimizer_name = str(optimizer_name).lower()
        if optimizer_name not in ("adamw", "sgd"):
            raise ValueError("optimizer_name must be adamw or sgd")
        if scheduler_name not in ("warmup_cosine", "reduce_on_plateau"):
            raise ValueError("unsupported scheduler_name")
        if scheduler_epochs is not None and (scheduler_name != "warmup_cosine" or scheduler_epochs < max_iter):
            raise ValueError("scheduler_epochs requires warmup_cosine and must cover all training epochs")
        if global_batch_size < 1 or global_batch_size % world_size:
            raise ValueError("global_batch_size must be positive and divisible by WORLD_SIZE")
        if max_iter < 1 or log_val < 1 or eval_every_steps < 1:
            raise ValueError("epochs, checkpoint interval, and validation interval must be positive")
        if world_size > 1 and mode != "vqvae":
            raise ValueError("DDP requires mode=vqvae; cvqvae has unsynchronized online codebook updates")
        local_batch = global_batch_size // world_size
        lengths = gather_records(len(dataloader))
        if not lengths[0] or len(set(lengths)) != 1:
            raise ValueError(f"all ranks must have the same non-zero step count: {lengths}")
        steps_per_epoch = lengths[0]
        total_steps = max_iter * steps_per_epoch
        if world_size > 1:
            sampler = getattr(dataloader, "batch_sampler", None)
            inner_sampler = getattr(dataloader, "sampler", None)
            if getattr(sampler, "num_replicas", None) != world_size and getattr(inner_sampler, "num_replicas", None) != world_size:
                raise ValueError("DDP training requires a rank-sharded sampler")
        self.to(device)
        trained_model = DistributedDataParallel(
            self, device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        ) if world_size > 1 else self
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay,
                                          betas=(0.9, 0.999), eps=1e-8)
            optimizer_settings = {"optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8}
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay,
                                        momentum=momentum)
            optimizer_settings = {"optimizer": "SGD", "momentum": momentum,
                                  "dampening": 0, "nesterov": False}
        schedule_steps = (scheduler_epochs or max_iter) * steps_per_epoch
        if scheduler_name == "warmup_cosine":
            scheduler, warmup_steps = make_warmup_cosine_scheduler(
                optimizer, schedule_steps, warmup_ratio, min_lr_ratio,
            )
            scheduler_settings = {"scheduler": "linear_warmup_cosine_decay",
                                  "scheduler_step_unit": "optimizer_update",
                                  "scheduler_total_steps": schedule_steps}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=plateau_factor, patience=plateau_patience,
            )
            warmup_steps = 0
            scheduler_settings = {"scheduler": "ReduceLROnPlateau", "scheduler_step_unit": "epoch",
                                  "plateau_factor": plateau_factor, "plateau_patience": plateau_patience}
        settings = {
            **optimizer_settings, **scheduler_settings,
            "learning_rate": lr, "weight_decay": weight_decay, "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps, "min_lr_ratio": min_lr_ratio,
            "global_batch_size": global_batch_size, "per_device_batch_size": local_batch,
            "world_size": world_size, "epochs": max_iter, "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps, "checkpoint_every_epochs": log_val,
            "eval_every_steps": eval_every_steps, "precision": "float32",
            "gradient_clip_norm": gradient_clip_norm,
        }
        output = Path(args.outdir)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        results = {key: [] for key in ("n_updates", "recon_loss", "embedding_loss", "train_loss", "perplexities", "eval_loss", "history")}
        start_epoch, global_step = 0, 0
        if resume is not None:
            start_epoch, global_step, results = restore_advanced_checkpoint(
                resume, self, optimizer, scheduler, settings, device,
            )
            if global_step != start_epoch * steps_per_epoch:
                raise ValueError("only epoch-boundary checkpoints can be resumed")
        records = gather_records({
            "rank": rank, "local_rank": int(os.environ.get("LOCAL_RANK", rank)),
            "device": str(device), "pid": os.getpid(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        })
        if rank == 0:
            print(json.dumps({"event": "advanced_fit_start", "ranks": records, "settings": settings,
                              "start_epoch": start_epoch, "global_step": global_step}), flush=True)
        started = time.monotonic()
        initial_step = global_step
        last_report = started
        state = {}

        def report(status, epoch, epoch_step, loss_value=None, rank_records=None):
            if rank != 0:
                return
            elapsed = time.monotonic() - started
            run_steps = global_step - initial_step
            step_seconds = elapsed / run_steps if run_steps else None
            state.update({
                "event": "advanced_fit_progress", "status": status,
                "epoch": epoch, "epochs": max_iter, "epoch_step": epoch_step,
                "steps_per_epoch": steps_per_epoch, "completed_steps": global_step,
                "total_steps": total_steps, "global_batch_size": global_batch_size,
                "world_size": world_size, "per_device_batch_size": local_batch,
                "cells_seen": global_step * global_batch_size,
                "elapsed_seconds": elapsed,
                "cells_per_second": run_steps * global_batch_size / max(elapsed, 1e-9),
                "eta_seconds": (total_steps-global_step) * step_seconds if step_seconds is not None else None,
                "epoch_eta_seconds": (steps_per_epoch-epoch_step) * step_seconds if step_seconds is not None else None,
                "learning_rate": optimizer.param_groups[0]["lr"], "loss": loss_value,
                "loss_scope": "rank0_current_batch",
                "rank0_cuda_allocated_mib": torch.cuda.memory_allocated(device)/1024**2 if device.type == "cuda" else 0,
                "rank0_cuda_reserved_mib": torch.cuda.memory_reserved(device)/1024**2 if device.type == "cuda" else 0,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "device": str(device), "pid": os.getpid(),
            })
            if rank_records is not None:
                state["rank_performance"] = rank_records
            temporary = output / "progress.json.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            os.replace(temporary, output / "progress.json")
            tqdm.write(json.dumps(state), file=sys.stdout)

        report("running", start_epoch+1, 0)
        try:
            with tqdm(total=total_steps, initial=global_step, unit="step", desc="VQVAE.advanced",
                      disable=rank != 0, dynamic_ncols=True, mininterval=2, file=sys.stdout) as progress:
                for epoch in range(start_epoch, max_iter):
                    for sampler in (getattr(dataloader, "batch_sampler", None), getattr(dataloader, "sampler", None)):
                        if hasattr(sampler, "set_epoch"):
                            sampler.set_epoch(epoch)
                    trained_model.train()
                    totals = torch.zeros(4, dtype=torch.float64, device=device)
                    validation_count = len(results["eval_loss"])
                    epoch_started = time.monotonic()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    loss_value = None
                    for i, batch in enumerate(dataloader):
                        correct_batch = torch.tensor(int(len(batch) == local_batch), device=device)
                        if world_size > 1:
                            dist.all_reduce(correct_batch, op=dist.ReduceOp.MIN)
                        if not correct_batch.item():
                            raise ValueError(f"every training batch on every rank must have {local_batch} cells; use drop_last=True")
                        x = batch.float().to(device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        values = trained_model(x, mode, device)
                        reconstruction = F.mse_loss(values[3], x)
                        embedding_loss = values[0]
                        loss = reconstruction + embedding_loss
                        finite = torch.isfinite(loss).to(torch.int32)
                        if world_size > 1:
                            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                        if not finite.item():
                            raise FloatingPointError(f"non-finite loss at epoch={epoch+1}, step={i+1}")
                        # DDP already averages gradients; do NOT divide loss by WORLD_SIZE.
                        loss.backward()
                        if gradient_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_norm)
                        lr_used = optimizer.param_groups[0]["lr"]
                        optimizer.step()
                        if scheduler_name == "warmup_cosine":
                            scheduler.step()
                        global_step += 1
                        loss_value = loss.item()
                        # Entropy/perplexity is nonlinear: combine token counts,
                        # rather than averaging the per-rank perplexities.
                        code_counts = values[7].detach().sum(dim=0)
                        if world_size > 1:
                            dist.all_reduce(code_counts)
                        code_probabilities = code_counts / code_counts.sum()
                        global_perplexity = torch.exp(-torch.sum(
                            code_probabilities * torch.log(code_probabilities + 1e-10)))
                        totals += torch.stack((reconstruction.detach().double()*len(x),
                                               embedding_loss.detach().double()*len(x),
                                               global_perplexity.double()*len(x),
                                               loss.new_tensor(len(x)).double()))
                        if (i+1) % eval_every_steps == 0:
                            results["eval_loss"].append(advanced_validation(trained_model, eval_loader, mode, device))
                        if rank == 0:
                            progress.set_description(f"VQVAE.advanced epoch {epoch+1}/{max_iter}", refresh=False)
                            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr_used:.2e}", epoch_step=f"{i+1}/{steps_per_epoch}", refresh=False)
                            progress.update(1)
                            if time.monotonic()-last_report >= 10 or global_step-initial_step == 1:
                                report("running", epoch+1, i+1, loss_value)
                                last_report = time.monotonic()
                    if len(results["eval_loss"]) == validation_count:
                        results["eval_loss"].append(advanced_validation(trained_model, eval_loader, mode, device))
                    if scheduler_name == "reduce_on_plateau":
                        # As in fit, once per epoch on the last periodic global
                        # validation value, not once per rank or validation batch.
                        scheduler.step(results["eval_loss"][-1])
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    epoch_seconds = time.monotonic() - epoch_started
                    rank_records = gather_records({
                        "rank": rank, "device": str(device), "steps": steps_per_epoch,
                        "cells": steps_per_epoch*local_batch, "seconds": epoch_seconds,
                        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else 0,
                        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device)/1024**2 if device.type == "cuda" else 0,
                    })
                    if world_size > 1:
                        dist.all_reduce(totals)
                    sync_error = parameter_sync_error(self)
                    if sync_error != 0:
                        raise AssertionError(f"DDP parameters diverged across ranks: {sync_error}")
                    denominator = totals[3].item()
                    recon_avg, embedding_avg, perplexity_avg = (totals[:3]/denominator).tolist()
                    results["n_updates"].append(epoch+1)
                    results["recon_loss"].append(recon_avg)
                    results["embedding_loss"].append(embedding_avg)
                    results["train_loss"].append(recon_avg+embedding_avg)
                    results["perplexities"].append(perplexity_avg)
                    record = {
                        "event": "advanced_fit_epoch", "epoch": epoch+1, "global_step": global_step,
                        "cells": int(denominator), "seconds": max(r["seconds"] for r in rank_records),
                        "reconstruction_loss": recon_avg, "embedding_loss": embedding_avg,
                        "total_loss": recon_avg+embedding_avg, "perplexity": perplexity_avg,
                        "validation_loss": results["eval_loss"][-1], "learning_rate": lr_used,
                        "next_learning_rate": optimizer.param_groups[0]["lr"],
                        "parameter_sync_max_abs_diff": sync_error, "rank_performance": rank_records,
                    }
                    results["history"].append(record)
                    if rank == 0:
                        tqdm.write(json.dumps(record), file=sys.stdout)
                        with (output/"metrics.jsonl").open("a") as handle:
                            handle.write(json.dumps(record)+"\n")
                    if save and ((epoch+1) % log_val == 0 or epoch+1 == max_iter):
                        save_advanced_checkpoint(self, optimizer, scheduler, results, args, epoch+1,
                                                 global_step, settings, device)
                    report("running", epoch+1, steps_per_epoch, loss_value, rank_records)
            report("complete", max_iter, steps_per_epoch, loss_value)
        except BaseException:
            report("failed", state.get("epoch", start_epoch+1), state.get("epoch_step", 0), state.get("loss"))
            raise
        return results


    def encode_batch(self, dataloader, mode, device='cuda', eval=True, transforms=None):
        if eval:
            self.eval()
        else:
            self.train()

        zq = []
        zoutput = []
        reconx = []
        for x in dataloader:
            x = x.float().to(device)
            _, _, _, x_hat, z_q, z_output, _, _, _ = self(x, mode, device)
            zq.append(z_q.detach().cpu())
            zoutput.append(z_output.detach().cpu())
            reconx.append(x_hat.cpu().detach().data)

        zq = torch.cat(zq)
        zoutput = torch.cat(zoutput)
        reconx = torch.cat(reconx)

        return zq, zoutput, reconx


    def get_codebook_index(self, dataloader, mode, device='cuda', eval=True, transforms=None):
        if eval:
            self.eval()
        else:
            self.train()

        indices = []
        for x in dataloader:
            x = x.float().to(device)
            _, _, _, _, _, _, _, min_encodings, min_encoding_indices = self(x, mode, device)

            min_encoding_indices = min_encoding_indices.view(x.shape[0], -1)
            indices.append(min_encoding_indices.detach().cpu())

        indices = torch.cat(indices)

        return indices


    def get_adata_codebook_index(self, adata, mode, device='cuda', eval=True, transforms=None):
        if eval:
            self.eval()
        else:
            self.train()

        batch_size = 5000
        _, dataloader = create_dataset_loader(
            adata=adata,
            batch_size=batch_size)

        indices = []
        zoutput = []
        reconx = []

        for x in dataloader:
            x = x.float().to(device)
            _, _, _, x_hat, _, z_output, _, min_encodings, min_encoding_indices = self(x, mode, device)

            min_encoding_indices = min_encoding_indices.view(x.shape[0], -1)
            indices.append(min_encoding_indices.detach().cpu())
            zoutput.append(z_output.detach().cpu())
            reconx.append(x_hat.cpu().detach().data)

        indices = torch.cat(indices)
        zoutput = torch.cat(zoutput)
        reconx = torch.cat(reconx)

        adata.obsm['code_index'] = indices.numpy()
        adata.obsm['latent'] = zoutput.numpy()
        adata.obsm['recon_x'] = reconx.numpy()

        return adata
