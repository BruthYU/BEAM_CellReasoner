#!/usr/bin/env python3
"""Compare actual DDP gradients and optimizer updates to a global-batch reference."""

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from lib_vqvae.advanced_training import make_warmup_cosine_scheduler
from lib_vqvae.vqvae import VQVAE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-world-size", default=4, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["warmup_cosine", "reduce_on_plateau"], default="warmup_cosine")
    cli = parser.parse_args()
    lr = 1e-4 if cli.optimizer == "sgd" else 3e-4
    torch.set_num_threads(1)
    local_rank = int(os.environ["LOCAL_RANK"])
    if cli.device == "cuda":
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if cli.device == "cuda" else torch.device("cpu")
    dist.init_process_group("nccl" if cli.device == "cuda" else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if world != cli.expected_world_size:
            raise AssertionError(f"expected {cli.expected_world_size} ranks, got {world}")
        torch.manual_seed(27)
        model = VQVAE(5, n=2, codebook_dim=2, n_codebooks=4,
                      encoder_hidden_dim=[6], decoder_hidden_dim=[6], dropout_rate=0).to(device)
        data = torch.rand(1536, 5, generator=torch.Generator().manual_seed(456))
        reference_parameters = torch.empty(sum(p.numel() for p in model.parameters()), device=device)
        reference_gradients = torch.empty_like(reference_parameters)
        if rank == 0:
            # Independent test-only global-batch calculation catches an extra
            # WORLD_SIZE loss division even when AdamW partly cancels its scale.
            reference = copy.deepcopy(model)
            optimizer = (torch.optim.SGD(reference.parameters(), lr=lr, momentum=.9, weight_decay=5e-4)
                         if cli.optimizer == "sgd" else
                         torch.optim.AdamW(reference.parameters(), lr=lr, weight_decay=5e-4))
            if cli.scheduler == "warmup_cosine":
                scheduler, _ = make_warmup_cosine_scheduler(optimizer, 6, 1/3, 0.1)
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=50)
            for _ in range(2):
                for batch in DataLoader(data, batch_size=512):
                    optimizer.zero_grad(set_to_none=True)
                    batch = batch.to(device)
                    values = reference(batch, "vqvae", device)
                    loss = torch.nn.functional.mse_loss(values[3], batch) + values[0]
                    loss.backward()
                    optimizer.step()
                    if cli.scheduler == "warmup_cosine":
                        scheduler.step()
                if cli.scheduler == "reduce_on_plateau":
                    scheduler.step(0.)  # patience=50: no decay in this two-epoch test.
            reference_parameters.copy_(torch.cat([p.detach().reshape(-1) for p in reference.parameters()]))
            reference_gradients.copy_(torch.cat([p.grad.detach().reshape(-1) for p in reference.parameters()]))
        dist.broadcast(reference_parameters, src=0)
        dist.broadcast(reference_gradients, src=0)
        sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=False)
        loader = DataLoader(data, batch_size=512//world, sampler=sampler, drop_last=True)
        model.fit_ddp_advanced(SimpleNamespace(outdir=str(cli.output)), loader, None, loader,
                               "vqvae", device, max_iter=2, global_batch_size=512,
                               warmup_ratio=1/3, log_val=1, lr=lr,
                               optimizer_name=cli.optimizer, scheduler_name=cli.scheduler)
        parameters = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        gradients = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])
        errors = torch.stack(((parameters-reference_parameters).abs().max(),
                              (gradients-reference_gradients).abs().max()))
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if errors[0] > 1e-5 or errors[1] > 5e-6:
            raise AssertionError(f"DDP differs from global-512 reference: parameter/gradient max errors={errors.tolist()}")
        if rank == 0:
            checkpoint = torch.load(cli.output/"epoch_2/model.pth", map_location="cpu", weights_only=False)
            expected_scheduler_steps = 2 if cli.scheduler == "reduce_on_plateau" else 6
            if checkpoint["global_step"] != 6 or checkpoint["scheduler"]["last_epoch"] != expected_scheduler_steps:
                raise AssertionError("DDP advanced scheduler stepped per rank rather than per optimizer update")
            if cli.optimizer == "adamw" and not all(int(s["step"]) == 6 for s in checkpoint["optimizer"]["state"].values()):
                raise AssertionError("AdamW optimizer step count mismatch")
            if cli.optimizer == "sgd" and not all("momentum_buffer" in s for s in checkpoint["optimizer"]["state"].values()):
                raise AssertionError("SGD momentum buffers missing")
            result = {"world_size": world, "global_batch_size": 512, "per_device_batch_size": 512//world,
                      "optimizer": cli.optimizer, "scheduler": cli.scheduler,
                      "optimizer_updates": 6, "scheduler_steps": expected_scheduler_steps,
                      "parameter_max_abs_error_vs_single": errors[0].item(),
                      "gradient_max_abs_error_vs_single": errors[1].item(), "passed": True}
            (cli.output/"equivalence.json").write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
