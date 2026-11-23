"""Trace one real held-out cell through the trained Stage 1 tokenizer."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import csv
import json

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from prepare_replogle_stage1 import ROOT, read_column
from sc_evae.models.factory import load_vae_from_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=64,
                        help="Index into the saved acceptance sample; 64 is the first held-out cell")
    parser.add_argument("--run-dir", default="outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401")
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit_dir = ROOT / "artifacts/replogle_stage1"
    raw_audit = json.loads((audit_dir / "raw_audit.json").read_text())
    loader_audit = json.loads((audit_dir / "loader_audit.json").read_text())
    symbols = (ROOT / "assets/replogle_scldm2k_gene_order.symbols.txt").read_text().splitlines()
    ensembl = (ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt").read_text().splitlines()
    with np.load(audit_dir / "checkpoint_sample.npz") as sample:
        global_row = int(sample["global_row"][args.sample_index])
        expression = sample["counts"][args.sample_index].copy()
        library_size = float(sample["library_size"][args.sample_index])
    offset = 0
    for file_index, metadata in enumerate(raw_audit["files_in_loader_order"]):
        if global_row < offset + metadata["kept_cells"]:
            break
        offset += metadata["kept_cells"]
    else:
        raise IndexError(global_row)
    source_path = ROOT / metadata["path"]
    source_row = int(np.flatnonzero(np.load(ROOT / metadata["kept_cell_mask"]))[global_row - offset])
    with h5py.File(source_path, "r") as source:
        var, obs = source["var"], source["obs"]
        native_genes = var[var.attrs.get("_index", "_index")].asstr()[:].tolist()
        positions = {gene: i for i, gene in enumerate(native_genes)}
        raw_full = source["X"][source_row].astype(np.float32)
        raw = raw_full[[positions[gene] for gene in ensembl]]
        perturbation = str(read_column(obs, "gene")[source_row])
        barcode = str(read_column(obs, obs.attrs.get("_index", "_index"))[source_row])
    target_sum = loader_audit["files"][file_index]["normalization_target_sum"]
    np.testing.assert_allclose(library_size, raw_full.sum())
    expected = np.log1p(raw * np.float32(target_sum / library_size))
    np.testing.assert_allclose(expression, expected, rtol=2e-6, atol=2e-6)
    split = json.loads((ROOT / "assets/scldm_replogle_test.json").read_text())
    heldout = metadata["cell_type"] in split["cell_type"] and perturbation in split["pert"]
    assert heldout, "This demonstration is intended to use a held-out cell"

    run_dir = ROOT / args.run_dir
    model = load_vae_from_run(str(run_dir), device="cpu")
    counts = torch.from_numpy(expression[None])
    with torch.no_grad():
        continuous = model.encode(counts)
        quantized = model.quantizer(continuous)
        restored_codes = model.quantizer.decode_indices(quantized.indices)
        torch.testing.assert_close(quantized.codes, restored_codes)
        reconstructed = model.decode(restored_codes).squeeze(0).numpy()
        reference = model.pseudo_sample({"counts": counts}).squeeze(0).numpy()
        np.testing.assert_allclose(reconstructed, reference, rtol=1e-5, atol=1e-6)
    indices = quantized.indices.squeeze(0).numpy()
    assert continuous.shape == (1, 64, 3) and indices.shape == (64,)
    assert indices.min() >= 0 and indices.max() < 512
    assert reconstructed.shape == (2000,) and np.isfinite(reconstructed).all()
    mse = float(np.mean((expression - reconstructed) ** 2))
    pearson = float(np.corrcoef(expression, reconstructed)[0, 1])
    # Choose the displayed genes by name, independently of reconstruction error.
    display_genes = [perturbation, "ISG15", "ENO1", "RPL11", "TFRC", "NCL", "GAPDH", "ACTB"]
    display_genes = list(dict.fromkeys(g for g in display_genes if g in symbols))
    display_indices = [symbols.index(g) for g in display_genes]
    example_rows = [{"gene": symbols[i], "ensembl": ensembl[i], "raw_umi": int(raw[i]),
                     "input_log1p": float(expression[i]),
                     "reconstructed_log1p": float(reconstructed[i])} for i in display_indices]
    output = run_dir / "analysis" / f"case_{metadata['cell_type']}_row{source_row}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "expression.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", "ensembl", "raw_umi", "input_log1p", "reconstructed_log1p"])
        writer.writerows(zip(symbols, ensembl, raw.tolist(), expression.tolist(), reconstructed.tolist()))
    np.savez_compressed(output / "tensors.npz", input_log1p=expression,
                        reconstructed_log1p=reconstructed, token_ids=indices,
                        continuous_latents=continuous.squeeze(0).numpy(),
                        quantized_codes=restored_codes.squeeze(0).numpy())
    report = {
        "selection": "First held-out cell in the seed-401 acceptance sample; not selected by reconstruction quality",
        "run_dir": str(run_dir), "source_file": str(source_path.resolve()),
        "source_row_zero_based": source_row, "global_row_zero_based": global_row,
        "barcode": barcode, "cell_type": metadata["cell_type"], "perturbation": perturbation,
        "split": "heldout", "native_gene_count": len(native_genes),
        "full_panel_raw_umi_total": library_size, "normalization_target_sum": target_sum,
        "preprocessing": "x_g = log1p(raw_umi_g * per_file_median_library / full_panel_raw_library); select ordered 2000 genes",
        "model_inputs": ["2000-gene expression vector"],
        "perturbation_and_cell_type_passed_to_model": False,
        "input_shape": list(counts.shape), "continuous_latent_shape": list(continuous.shape),
        "token_shape": [1, len(indices)], "token_ids": indices.tolist(),
        "first_token_continuous_latent": continuous[0, 0].tolist(),
        "first_token_quantized_code": restored_codes[0, 0].tolist(),
        "first_token_id": int(indices[0]),
        "output_shape": [1, len(reconstructed)],
        "output_space": "Continuous log1p normalized expression, same ordered 2000 genes as input; not raw integer UMI",
        "mse_over_2000_genes": mse, "pearson_over_2000_genes": pearson,
        "pearson_scope": "Within this one cell across genes; not a perturbation-prediction metric",
        "negative_reconstruction_values": int((reconstructed < 0).sum()),
        "display_genes": example_rows, "inference_device": "cpu", "precision": "fp32",
        "model_sha256": json.loads((audit_dir / f"{run_dir.name}-checkpoint.json").read_text())["model_sha256"],
    }
    (output / "case.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(13, 7.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 3])
    token_ax = fig.add_subplot(grid[0, :])
    # Wrap the 64 ordered positions into four rows solely for display.
    token_ax.imshow(indices.reshape(4, 16), cmap="Blues", vmin=0, vmax=511, aspect="auto")
    for position, token in enumerate(indices):
        token_ax.text(position % 16, position // 16, str(token), ha="center", va="center",
                      color="white" if token > 275 else "#17354a", fontsize=9)
    token_ax.set_xticks([])
    token_ax.set_yticks(range(4), ["Positions 1–16", "17–32", "33–48", "49–64"])
    token_ax.set_title("64 FSQ token IDs (each in 0–511)", loc="left", fontweight="bold")
    for spine in token_ax.spines.values():
        spine.set_visible(False)
    scatter = fig.add_subplot(grid[1, 0])
    scatter.scatter(expression, reconstructed, s=9, color="#2563a6", alpha=.35, edgecolors="none", rasterized=True)
    lower = min(0., float(reconstructed.min()))
    upper = max(float(expression.max()), float(reconstructed.max())) * 1.04
    scatter.plot([lower, upper], [lower, upper], "--", color="#777777", lw=1)
    scatter.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Observed log1p expression",
                ylabel="Reconstructed log1p expression")
    scatter.set_title(f"All 2,000 genes · MSE {mse:.4f} · Pearson {pearson:.3f}", loc="left", fontweight="bold")
    bars = fig.add_subplot(grid[1, 1])
    x = np.arange(len(display_indices))
    bars.bar(x - .19, expression[display_indices], width=.38, label="Observed", color="#9ca3af")
    bars.bar(x + .19, reconstructed[display_indices], width=.38, label="Reconstructed", color="#2563a6")
    bars.set_xticks(x, display_genes, rotation=35, ha="right")
    bars.set_ylabel("Log1p normalized expression")
    bars.set_title("Example genes (fixed by name)", loc="left", fontweight="bold")
    bars.legend(frameon=False)
    fig.suptitle(f"Real held-out cell: HepG2 · {perturbation} CRISPRi\n"
                 "2,000 expression values → 64 FSQ tokens → 2,000 reconstructed values",
                 fontsize=14, fontweight="bold")
    fig.savefig(output / "cell_tokenizer_case.png", dpi=220)
    fig.savefig(output / "cell_tokenizer_case.pdf", dpi=220)
    plt.close(fig)
    print(json.dumps({"output_dir": str(output), **report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
