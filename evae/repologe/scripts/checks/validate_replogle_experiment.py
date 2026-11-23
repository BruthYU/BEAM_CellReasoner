"""Validate configurable MSE/NB experiments using the original dataset/model APIs."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import hashlib
import json
import logging
from pathlib import Path

import draccus
import h5py
import numpy as np
import torch

from sc_evae.config.training import TrainConfig
from sc_evae.config.models import NBHeadConfig, MSEHeadConfig
from sc_evae.models.autoregressive import AutoregressiveModel
from sc_evae.models.factory import make_model, load_vae_from_run, load_weights_from_checkpoint
from sc_evae.training.data_loader import PerturbationDataset
from sc_evae.training.train_utils import latest_checkpoint, resolve_train_val_split

ROOT = Path(__file__).resolve().parents[2]


def config(path):
    with (ROOT / path).open() as handle:
        return draccus.load(TrainConfig, handle)


def digest_state(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sample_batch(head, device, with_embeddings=False):
    # Reuse the SAME sampled cell identities, with the requested head's input units.
    with np.load(ROOT / "artifacts/replogle_stage1/checkpoint_sample.npz") as data:
        chosen = np.array([0, 1, 64, 65])
        arrays = {k: data[k][chosen].copy() for k in data.files}
    if head == "nb":
        genes = (ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt").read_text().splitlines()
        audit = json.loads((ROOT / "artifacts/replogle_stage1/raw_audit.json").read_text())
        rows = []
        for global_row in arrays["global_row"]:
            offset = 0
            for file in audit["files_in_loader_order"]:
                if global_row < offset + file["kept_cells"]:
                    break
                offset += file["kept_cells"]
            physical = int(np.flatnonzero(np.load(ROOT / file["kept_cell_mask"]))[global_row - offset])
            with h5py.File(ROOT / file["path"], "r") as source:
                var = source["var"]
                native = var[var.attrs.get("_index", "_index")].asstr()[:]
                lookup = {gene: i for i, gene in enumerate(native)}
                raw = source["X"][physical].astype(np.float32)
                rows.append(raw[[lookup[gene] for gene in genes]])
        arrays["counts"] = np.stack(rows)
        assert np.all(arrays["counts"] >= 0) and np.array_equal(arrays["counts"], np.floor(arrays["counts"]))
    batch = {k: torch.from_numpy(arrays[k]).to(device)
             for k in ["counts", "library_size", "pert_idx", "cell_type_idx"]}
    if with_embeddings:
        audit = json.loads((ROOT / "artifacts/replogle_stage2/embedding_audit.json").read_text())
        features = torch.load(audit["path"], map_location="cpu", weights_only=True)
        batch["pert_emb"] = torch.stack([features.get(audit["required_labels"][int(i)],
            torch.zeros(audit["feature_dim"])).float() for i in batch["pert_idx"].cpu()]).to(device)
    return batch, arrays["global_row"]


def resolve_model(cfg, counts, prior=False):
    cfg.model.set_attributes_from_dataset(num_genes=2000, count_max=int(counts.max()),
        num_cell_types=4, num_perts=2396, pert_emb_dim=5120 if prior else None)
    cfg.model.validate()


def finite_gradients(model):
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def check_tokens(vae, counts, parameters):
    with torch.no_grad():
        q = vae.quantizer(vae.encode(counts))
        repeat = vae.quantizer(vae.encode(counts))
        assert torch.equal(q.indices, repeat.indices)
        assert q.indices.shape == (len(counts), parameters["num_latent_tokens"])
        assert q.indices.min() >= 0 and q.indices.max() < parameters["codebook_size"]
        torch.testing.assert_close(vae.quantizer.decode_indices(q.indices).float(), q.codes.float())
    return q.indices


def check_expression(value, head):
    assert value.shape[-1] == 2000 and torch.isfinite(value).all()
    if head == "nb":
        assert (value >= 0).all() and torch.equal(value, value.round())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True)
    p.add_argument("--phase", choices=["cpu", "data", "stage1", "stage2"], default="cpu")
    p.add_argument("--run")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    spec = json.loads((ROOT / args.spec).read_text())
    params = spec["parameters"]
    torch.manual_seed(params["stage1_seed"])
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available()
    logging.basicConfig(level=logging.INFO)
    head = params["head"]
    report = {"status": "passed", "experiment": spec["experiment"], "head": head,
              "phase": args.phase, "device": str(device), "parameters": params}
    artifacts = ROOT / spec["artifacts"]
    if args.phase == "data":
        cfg = config(spec["config_root"] + "/stage1.yaml")
        dataset = PerturbationDataset(cfg.dataset, logger=logging.getLogger(__name__))
        train, heldout = resolve_train_val_split(dataset, cfg.dataset, cfg.seed)
        original = json.loads((ROOT / "artifacts/replogle_stage1/loader_audit.json").read_text())
        for name, view in [("train", train), ("heldout", heldout)]:
            assert hashlib.sha256(view._indices.tobytes()).hexdigest() == original[name + "_indices_sha256"]
        assert dataset.num_genes == 2000 and len(dataset) == 966728
        batch, rows = sample_batch(head, "cpu")
        np.testing.assert_allclose(dataset.counts[rows], batch["counts"].numpy(), rtol=2e-6, atol=2e-6)
        for start in range(0, len(dataset), 8192):
            block = dataset.counts[start:start + 8192]
            assert np.isfinite(block).all() and (block >= 0).all()
            if head == "nb":
                assert np.array_equal(block, np.floor(block))
        report.update(train_cells=len(train), heldout_cells=len(heldout), genes=2000,
            split_identical_to_existing_mse_run=True, input_space="raw_umi" if head == "nb" else "normalized_log1p")
        name = "data_acceptance.json"
    else:
        batch, _ = sample_batch(head, device, with_embeddings=True)
        cfg1 = config(spec["config_root"] + "/stage1.yaml")
        resolve_model(cfg1, batch["counts"])
        assert cfg1.model.quantizer.codebook_size == params["codebook_size"]
        losses = []
        if args.phase == "cpu":
            # One real cell keeps the full decoder backward pass within login RAM.
            stats = json.loads((ROOT / "outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401/dataset_stats.json").read_text())
            vae = make_model(cfg1.model, log_lib_prior_mu=stats["log_lib_mean"], log_lib_prior_std=stats["log_lib_std"]).to(device)
            vae.train()
            optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg1.learning_rate)
            loss, _ = vae({"counts": batch["counts"][:1]})
            assert torch.isfinite(loss)
            loss.backward()
            finite_gradients(vae)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg1.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            vae.eval()
        elif args.phase == "stage1":
            vae = load_vae_from_run(str(ROOT / args.run), device=device)
        else:
            vae = None
        if args.phase in ["cpu", "stage1"]:
            assert isinstance(vae.config.output_head, NBHeadConfig if head == "nb" else MSEHeadConfig)
            assert all(torch.isfinite(value).all() for value in vae.parameters())
            ids = check_tokens(vae, batch["counts"][:1], params)
            with torch.no_grad():
                value = vae.pseudo_sample({"counts": batch["counts"][:1]})
                check_expression(value, head)
                loss, _ = vae({"counts": batch["counts"][:1]})
                assert torch.isfinite(loss)
            report.update(stage1_loss=float(loss), stage1_preflight_update_losses=losses,
                stage1_token_shape=list(ids.shape), reconstructed_shape=list(value.shape),
                reconstruction_output_space="sampled_integer_umi" if head == "nb" else "normalized_log1p")
        if args.phase in ["cpu", "stage2"]:
            cfg2 = config(str(Path(args.run) / "config.yaml") if args.phase == "stage2" else spec["config_root"] + "/stage2.yaml")
            resolve_model(cfg2, batch["counts"], prior=True)
            if args.phase == "cpu":
                model = AutoregressiveModel(cfg2.model, vae=vae).to(device)
            else:
                model = make_model(cfg2.model)
                before = digest_state(model.vae)
                checkpoint = latest_checkpoint(str(ROOT / args.run / "checkpoints"))
                assert checkpoint is not None
                load_weights_from_checkpoint(model, checkpoint)
                assert digest_state(model.vae) == before
                model.to(device)
            frozen = digest_state(model.vae)
            model.train()
            assert not model.vae.training and all(not p.requires_grad for p in model.vae.parameters())
            target = check_tokens(model.vae, batch["counts"][:1], params)
            if args.phase == "cpu":
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg2.learning_rate)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = model(batch)
                    assert torch.isfinite(loss)
                    loss.backward()
                    finite_gradients(model)
                    assert all(p.grad is None for p in model.vae.parameters())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg2.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            assert digest_state(model.vae) == frozen
            model.eval()
            with torch.no_grad():
                loss, _ = model(batch)
                assert torch.isfinite(loss)
                tokens = model.sample(batch_size=1, pert_emb=batch["pert_emb"][:1],
                    cell_type_idx=batch["cell_type_idx"][:1], device=device)
                assert tokens.shape == (1, params["num_latent_tokens"])
                assert tokens.min() >= 0 and tokens.max() < params["codebook_size"]
                expression = model.vae.decode_observed(model.vae.quantizer.decode_indices(tokens), None,
                    batch["counts"][:1].sum(-1, keepdim=True))
                check_expression(expression, head)
            report.update(stage2_token_cross_entropy=float(loss), frozen_vae_weights_unchanged=True,
                tokenizer_eval_during_parent_train=True, deterministic_target_tokens=True,
                generated_token_shape=list(tokens.shape), generated_expression_shape=list(expression.shape),
                condition_only_ancestral_generation=True)
            assert all(torch.isfinite(value).all() for value in model.parameters())
        if args.phase != "cpu":
            saved_cfg = config(str(Path(args.run) / "config.yaml"))
            checkpoint = latest_checkpoint(str(ROOT / args.run / "checkpoints"))
            assert checkpoint and Path(checkpoint).name == f"step_{saved_cfg.n_iters:07d}_final"
            for required in ["model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"]:
                assert (Path(checkpoint) / required).is_file()
            assert (ROOT / args.run / "dataset_stats.json").is_file()
            deg = ROOT / saved_cfg.dataset.deg_table_path if saved_cfg.dataset.deg_table_path else ROOT / args.run / "deg_table.pt"
            assert deg.is_file()
            report.update(checkpoint=checkpoint, model_sha256=hashlib.sha256((Path(checkpoint) / "model.safetensors").read_bytes()).hexdigest())
        name = "cpu_preflight.json" if args.phase == "cpu" else Path(args.run).name + "-checkpoint.json"
    destination = artifacts / name
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
