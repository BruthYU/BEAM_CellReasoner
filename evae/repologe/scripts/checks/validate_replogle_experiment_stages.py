"""Stage-specific checks; use the same original training/model implementations."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import hashlib
import json
from pathlib import Path

import torch

import validate_replogle_experiment as original

ROOT = original.ROOT


def check_saved_run(run, cfg):
    checkpoint = original.latest_checkpoint(str(ROOT / run / "checkpoints"))
    assert checkpoint and Path(checkpoint).name == f"step_{cfg.n_iters:07d}_final"
    for name in ["model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"]:
        assert (Path(checkpoint) / name).is_file()
    assert (ROOT / run / "dataset_stats.json").is_file()
    deg = ROOT / cfg.dataset.deg_table_path if cfg.dataset.deg_table_path else ROOT / run / "deg_table.pt"
    assert deg.is_file()
    return checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True)
    p.add_argument("--phase", choices=["cpu", "data", "stage1", "stage2"], default="cpu")
    p.add_argument("--run")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    # Existing data and AR checkpoint checks are unchanged.
    if args.phase in ["data", "stage2"]:
        original.main()
        return
    torch.set_num_threads(2)
    spec = json.loads((ROOT / args.spec).read_text())
    params, stage = spec["parameters"], spec["stage"]
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available()
    torch.manual_seed(params["stage1_seed"] if stage != "stage2" else params["stage2_seed"])
    head = params["head"]
    batch, _ = original.sample_batch(head, device, with_embeddings=stage != "stage1" and args.phase == "cpu")
    report = {"status": "passed", "experiment": spec["experiment"], "stage": stage,
              "phase": args.phase, "head": head, "device": str(device), "stages_checked": []}
    if args.phase == "stage1":
        cfg = original.config(str(Path(args.run) / "config.yaml"))
        vae = original.load_vae_from_run(str(ROOT / args.run), device=device)
        checkpoint = check_saved_run(args.run, cfg)
        report.update(checkpoint=checkpoint,
            model_sha256=hashlib.sha256((Path(checkpoint) / "model.safetensors").read_bytes()).hexdigest())
    elif stage == "stage2":
        # The frozen model is loaded from the user's trained run; no Stage 1 update.
        vae = original.load_vae_from_run(str(ROOT / spec["stage1_run"]), device=device)
    else:
        cfg = original.config(spec["config_root"] + "/stage1.yaml")
        original.resolve_model(cfg, batch["counts"])
        # Approximate prior only for this discarded CPU probe. Training computes
        # its own dataset statistics through the original scripts/train.py.
        log_lib = batch["library_size"].float().log()
        vae = original.make_model(cfg.model, log_lib_prior_mu=float(log_lib.mean()),
                                  log_lib_prior_std=float(log_lib.std())).to(device)
        optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.learning_rate)
        vae.train()
        loss, _ = vae({"counts": batch["counts"][:1]})
        assert torch.isfinite(loss)
        loss.backward()
        original.finite_gradients(vae)
        torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report["stage1_probe_update_loss"] = float(loss.detach())
        report["stages_checked"].append("stage1")
    vae.eval()
    assert isinstance(vae.config.output_head, original.NBHeadConfig if head == "nb" else original.MSEHeadConfig)
    assert all(torch.isfinite(v).all() for v in vae.parameters())
    tokens = original.check_tokens(vae, batch["counts"][:1], params)
    report["target_token_shape"] = list(tokens.shape)
    if args.phase == "stage1" or stage != "stage2":
        with torch.no_grad():
            loss, _ = vae({"counts": batch["counts"][:1]})
            assert torch.isfinite(loss)
            prediction = vae.pseudo_sample({"counts": batch["counts"][:1]})
            original.check_expression(prediction, head)
        report.update(stage1_loss=float(loss), reconstructed_expression_shape=list(prediction.shape))
    if args.phase == "cpu" and stage != "stage1":
        cfg = original.config(spec["config_root"] + "/stage2.yaml")
        original.resolve_model(cfg, batch["counts"], prior=True)
        if stage == "stage2":
            assert original.digest_state(vae)
            report["reused_vae_run"] = spec["stage1_run"]
            assert hashlib.sha256((ROOT / spec["stage1_run"] / "checkpoints" /
                spec["source_vae"]["checkpoint"] / "model.safetensors").read_bytes()).hexdigest() == spec["source_vae"]["model_sha256"]
        model = original.AutoregressiveModel(cfg.model, vae=vae).to(device)
        frozen = original.digest_state(model.vae)
        model.train()
        assert not model.vae.training and all(not p.requires_grad for p in model.vae.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch)
            assert torch.isfinite(loss)
            loss.backward()
            original.finite_gradients(model)
            assert all(p.grad is None for p in model.vae.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert original.digest_state(model.vae) == frozen
        model.eval()
        with torch.no_grad():
            generated = model.sample(batch_size=1, pert_emb=batch["pert_emb"][:1],
                                     cell_type_idx=batch["cell_type_idx"][:1], device=device)
            assert generated.shape == (1, params["num_latent_tokens"])
            assert generated.min() >= 0 and generated.max() < params["codebook_size"]
            prediction = model.vae.decode_observed(model.vae.quantizer.decode_indices(generated),
                None, batch["counts"][:1].sum(-1, keepdim=True))
            original.check_expression(prediction, head)
        report.update(stage2_probe_cross_entropy=float(loss.detach()), frozen_vae_weights_unchanged=True,
            tokenizer_eval_during_parent_train=True, generated_token_shape=list(generated.shape),
            generated_expression_shape=list(prediction.shape), condition_only_ancestral_generation=True)
        report["stages_checked"].append("stage2")
    name = "cpu_preflight.json" if args.phase == "cpu" else Path(args.run).name + "-checkpoint.json"
    (ROOT / spec["artifacts"] / name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
