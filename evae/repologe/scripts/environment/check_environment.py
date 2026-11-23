"""Check installed dependencies, H5AD I/O, and a small VAE training step."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import importlib
import importlib.metadata
import json
import pkgutil
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch

import sc_evae
from sc_evae.config.models import FSQConfig, TransformerConfig, sc_evaeConfig
from sc_evae.metrics.distribution import wasserstein2_pca
from sc_evae.models.vae import sc_evae as ExpressionVAE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but no usable GPU is visible")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    torch.manual_seed(42)

    # Import every project module to catch missing runtime dependencies.
    modules = list(pkgutil.walk_packages(sc_evae.__path__, "sc_evae."))
    for module in modules:
        importlib.import_module(module.name)

    counts = np.random.default_rng(42).poisson(2, size=(8, 16)).astype("float32")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cells.h5ad"
        ad.AnnData(counts).write_h5ad(path)
        cells = ad.read_h5ad(path)
        np.testing.assert_array_equal(cells.X, counts)
        sc.pp.normalize_total(cells, target_sum=10000)
        sc.pp.log1p(cells)
        data = torch.tensor(cells.X, device=device)

    def backbone():
        return TransformerConfig(
            d_model=32, nhead=4, num_layers=1, dim_feedforward=64,
            attn_dropout=0.0,
        )

    config = sc_evaeConfig(
        num_genes=16, is_count_discrete=False, num_latent_tokens=4,
        quantizer=FSQConfig(codebook_size=64),
        encoder=backbone(), decoder=backbone(),
    )
    config.validate()
    model = ExpressionVAE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, _ = model({"counts": data})
    assert torch.isfinite(loss), loss
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer.step()
    model.eval()
    prediction = model.pseudo_sample({"counts": data})
    assert prediction.shape == data.shape and torch.isfinite(prediction).all()
    distance = wasserstein2_pca(prediction, data, n_components=3)
    assert np.isfinite(distance)

    versions = {
        name: importlib.metadata.version(name)
        for name in ("sc_evae", "torch", "numpy", "scanpy", "anndata", "accelerate")
    }
    print(json.dumps({
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "project_modules_imported": len(modules),
        "vae_loss": loss.item(),
        "wasserstein_distance": distance,
        "h5ad_roundtrip": "passed",
        "vae_forward_backward_optimizer_prediction": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
