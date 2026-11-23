"""Check the original AR prior, frozen tokenizer, and condition-only generation."""

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
import numpy as np
import torch

from sc_evae.config.training import TrainConfig
from sc_evae.config.models import AutoregressiveModelConfig
from sc_evae.models.factory import make_model, load_weights_from_checkpoint
from sc_evae.training.data_loader import PerturbationDataset
from sc_evae.training.train_utils import latest_checkpoint, resolve_train_val_split

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/replogle_stage2"


def load_config(path):
    with (ROOT / path).open() as handle:
        return draccus.load(TrainConfig, handle)


def write_report(name, report):
    report["status"] = "passed"
    destination = ARTIFACTS / name
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def validate_data(cfg):
    previous = json.loads((ROOT / "artifacts/replogle_stage1/loader_audit.json").read_text())
    embeddings_report = json.loads((ARTIFACTS / "embedding_audit.json").read_text())
    ds = PerturbationDataset(config=cfg.dataset, logger=logging.getLogger(__name__))
    train, heldout = resolve_train_val_split(ds, cfg.dataset, cfg.seed)
    assert len(train) == previous["train_cells"] and len(heldout) == previous["heldout_cells"]
    assert ds.num_genes == 2000 and ds.cell_type_to_idx == previous["cell_type_to_idx"]
    assert ds.unique_perts == embeddings_report["required_labels"]
    assert ds.pert_emb_dim == embeddings_report["feature_dim"] and torch.isfinite(ds.pert_emb).all()
    for name, view in [("train", train), ("heldout", heldout)]:
        assert hashlib.sha256(view._indices.tobytes()).hexdigest() == previous[f"{name}_indices_sha256"]
    with np.load(ROOT / "artifacts/replogle_stage1/checkpoint_sample.npz") as sample:
        rows = sample["global_row"]
        np.testing.assert_allclose(ds.counts[rows], sample["counts"], rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(ds._pert_indices[rows], sample["pert_idx"])
        np.testing.assert_array_equal(ds._cell_type_indices[rows], sample["cell_type_idx"])
    write_report("data_acceptance.json", {"train_cells": len(train), "heldout_cells": len(heldout),
        "genes": ds.num_genes, "pert_emb_dim": ds.pert_emb_dim,
        "split_indices_identical_to_stage1": True, "sample_expression_identical_to_stage1": True,
        "cell_type_to_idx": ds.cell_type_to_idx, "perturbations_including_control": ds.num_perts,
        "missing_embedding_labels": embeddings_report["missing_labels"],
        "heldout_missing_or_zero": embeddings_report["heldout_missing_or_zero"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/local/replogle-stage2-ar-fsq-mse.yaml")
    parser.add_argument("--checkpoint-run")
    parser.add_argument("--data", action="store_true", help="Full eager-loader acceptance; worker memory required")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(420)
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available()
    cfg = load_config(str(Path(args.checkpoint_run) / "config.yaml") if args.checkpoint_run else args.config)
    logging.basicConfig(level=logging.INFO)
    if args.data:
        validate_data(cfg)
        return
    source_run = ROOT / cfg.model.vae_path
    stage1 = json.loads((ROOT / "artifacts/replogle_stage1/completion.json").read_text())
    assert stage1["status"] == "complete"
    source_checkpoint = latest_checkpoint(str(source_run / "checkpoints"))
    assert source_checkpoint is not None
    source_weights = Path(source_checkpoint) / "model.safetensors"
    assert hashlib.sha256(source_weights.read_bytes()).hexdigest() == stage1["model_sha256"]
    embeddings_report = json.loads((ARTIFACTS / "embedding_audit.json").read_text())
    assert embeddings_report["status"] == "passed"
    cfg.model.set_attributes_from_dataset(num_genes=2000,
        pert_emb_dim=embeddings_report["feature_dim"], num_cell_types=4,
        num_perts=embeddings_report["dataset_labels"])
    cfg.model.validate()
    assert isinstance(cfg.model, AutoregressiveModelConfig)
    assert cfg.model.do_pert_cond and cfg.model.do_celltype_cond
    assert cfg.model.pert_cond_source == "projected"
    assert cfg.dataset.split_path == "assets/scldm_replogle_test.json"
    assert cfg.dataset.gene_order_path == "assets/replogle_scldm2k_gene_order.ensembl.txt"
    assert (ROOT / cfg.dataset.deg_table_path).is_file()
    embeddings = torch.load(ROOT / cfg.dataset.pert_mapping_path, map_location="cpu", weights_only=True)
    labels = embeddings_report["required_labels"]
    selected = np.array([0, 1, 64, 65])
    with np.load(ROOT / "artifacts/replogle_stage1/checkpoint_sample.npz") as data:
        batch = {key: torch.from_numpy(data[key][selected].copy()).to(device)
                 for key in ["counts", "library_size", "pert_idx", "cell_type_idx"]}
    batch["pert_emb"] = torch.stack([embeddings.get(labels[int(i)],
        torch.zeros(cfg.model.pert_emb_dim)).float() for i in batch["pert_idx"].cpu()]).to(device)
    assert torch.isfinite(batch["pert_emb"]).all()
    del embeddings
    model = make_model(cfg.model)
    expected_vae = {key: value.clone() for key, value in model.vae.state_dict().items()}
    checkpoint = None
    if args.checkpoint_run:
        run = ROOT / args.checkpoint_run
        checkpoint = latest_checkpoint(str(run / "checkpoints"))
        assert checkpoint is not None and Path(checkpoint).name == f"step_{cfg.n_iters:07d}_final"
        for name in ["model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"]:
            assert (Path(checkpoint) / name).is_file(), name
        load_weights_from_checkpoint(model, checkpoint)
        assert all(torch.equal(expected_vae[k], value) for k, value in model.vae.state_dict().items())
    model = model.to(device)
    assert model.bos_token_id == 512
    assert model.output_head.proj.out_features == 512
    model.train()
    assert model.training and not model.vae.training
    assert all(not parameter.requires_grad for parameter in model.vae.parameters())
    with torch.no_grad():
        targets = model.vae.quantizer(model.vae.encode(batch["counts"])).indices
        repeat = model.vae.quantizer(model.vae.encode(batch["counts"])).indices
    assert torch.equal(targets, repeat), "Frozen tokenizer targets must be deterministic"
    assert targets.shape == (4, 64) and targets.min() >= 0 and targets.max() < 512
    losses = []
    if not args.checkpoint_run:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        for _ in range(3):
            optimizer.zero_grad()
            loss, _ = model(batch)
            assert torch.isfinite(loss)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert all(p.grad is None for p in model.vae.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        assert all(torch.equal(expected_vae[k], value.cpu()) for k, value in model.vae.state_dict().items())
    model.eval()
    with torch.no_grad():
        loss, _ = model(batch)
        assert torch.isfinite(loss)
        # True ancestral generation: only perturbation and cell type are passed.
        generated = model.sample(batch_size=2, pert_emb=batch["pert_emb"][:2],
            cell_type_idx=batch["cell_type_idx"][:2], device=device)
        assert generated.shape == (2, 64)
        assert generated.dtype == torch.int64 and generated.min() >= 0 and generated.max() < 512
        reconstructed = model.vae.decode(model.vae.quantizer.decode_indices(generated))
        assert reconstructed.shape == (2, 2000) and torch.isfinite(reconstructed).all()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    report = {"device": str(device), "model": "autoregressive", "checkpoint": checkpoint,
        "source_vae_run": str(source_run), "source_vae_sha256": stage1["model_sha256"],
        "source_vae_weights_unchanged": True, "vae_eval_during_parent_train": True,
        "deterministic_target_tokens": True, "target_token_shape": list(targets.shape),
        "bos_token_id": model.bos_token_id, "predicted_vocab_size": 512,
        "pert_emb_dim": cfg.model.pert_emb_dim, "num_cell_types": 4,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "cpu_preflight_update_losses": losses, "sample_token_cross_entropy": float(loss),
        "condition_only_ancestral_generation": True,
        "generated_token_shape": list(generated.shape), "decoded_expression_shape": list(reconstructed.shape),
        "generated_token_range": [int(generated.min()), int(generated.max())]}
    name = f"{Path(args.checkpoint_run).name}-checkpoint.json" if args.checkpoint_run else "model_preflight.json"
    write_report(name, report)


if __name__ == "__main__":
    main()
