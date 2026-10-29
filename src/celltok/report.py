"""Generic tokenizer evaluation tables and epoch curves (no experiment dates)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("mse", "pcc", "acs_cellwise", "acs_genewise", "acs_global", "mmd_rbf_biased_squared",
           "mmd_half_biased_squared", "reconstruction_clustering.ari", "reconstruction_clustering.nmi",
           "reconstruction_clustering.ami", "latent_clustering.ari", "latent_clustering.nmi",
           "latent_clustering.ami", "codebook_usage.active_codes", "codebook_usage.code_perplexity")


def assert_same_protocol(results):
    if not results:
        raise ValueError("No evaluation results supplied")
    fields = ("manifest", "split", "cells", "normalization", "protocol")
    for result in results[1:]:
        for field in fields:
            if result.get(field) != results[0].get(field):
                raise ValueError(f"Different evaluation protocols: {field}")
    return {field: results[0].get(field) for field in fields}


def metric(value, key):
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(evaluations, output, paper=None):
    names = list(evaluations)
    results = [json.loads(path.read_text()) for path in evaluations.values()]
    protocol = assert_same_protocol(results)
    rows = [{"model": name, **{key: metric(result["metrics"], key) for key in METRICS}}
            for name, result in zip(names, results)]
    write_csv(output / "comparison.csv", rows)
    deltas = [{"model": row["model"], **{key: row[key]-rows[0][key] for key in METRICS
               if row[key] is not None and rows[0][key] is not None}} for row in rows[1:]]
    summary = {"protocol": protocol, "rows": rows, "baseline": names[0], "deltas_versus_baseline": deltas,
               "sources": [{"label": name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for name, path in evaluations.items()],
               "notes": ["Identical protocol metadata required, including cell indices and software versions.",
                         "ACS axis and MMD kernel/bandwidth/scaling are not fully calibrated to the paper.",
                         "Code utilization/perplexity is diagnostic, not a direct biological-quality score."]}
    if paper:
        summary["paper_reference"] = paper
    (output / "comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
    panels = (("mse", "MSE (lower is better)"), ("pcc", "PCC"),
              ("reconstruction_clustering.ari", "Reconstruction ARI"), ("reconstruction_clustering.nmi", "Reconstruction NMI"),
              ("latent_clustering.ari", "Quantized latent ARI"), ("latent_clustering.nmi", "Quantized latent NMI"),
              ("mmd_rbf_biased_squared", "Biased RBF MMD squared"), ("codebook_usage.code_perplexity", "Code usage perplexity"))
    columns = 4
    if all("reconstruction_clustering" not in result["metrics"] for result in results):
        panels = (("mse", "MSE (lower is better)"), ("pcc", "Mean cellwise PCC"),
                  ("acs_cellwise", "Mean cellwise ACS"), ("acs_genewise", "Mean genewise ACS"),
                  ("acs_global", "Global ACS"))
        columns = 3
        summary["notes"] = ["All reconstruction metrics use the entire recorded evaluation cohort.",
                            "MMD and clustering are absent from this reconstruction-only comparison.",
                            "ACS axis must be kept explicit when comparing against the paper."]
        (output / "comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
    fig, axes = plt.subplots(2, columns, figsize=(4*columns, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, panels):
        values = [row[key] for row in rows]
        if any(value is None for value in values):
            ax.set_visible(False)
            continue
        bars = ax.bar(names, values)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
        if paper:
            reference = paper.get(key) if "." not in key else metric(paper, key.replace("_clustering", ""))
            if reference is not None and key not in ("mmd_rbf_biased_squared", "codebook_usage.code_perplexity"):
                ax.axhline(reference, ls="--", color="red", lw=1, label="Paper reference")
                ax.legend(fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.margins(y=.15)
        ax.grid(axis="y", alpha=.2)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"Same {protocol['cells']:,} evaluation cells, preprocessing and metric protocol")
    save_figure(fig, output / "comparison")
    return summary


def load_history(path):
    source = path / "metrics.jsonl"
    lines = source.read_text().splitlines(keepends=True)
    rows = []
    for number, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # The last log line may still be being appended by a live run.
            if number == len(lines)-1 and not line.endswith("\n"):
                continue
            raise
        if "reconstruction_loss" in record and "epoch" in record:
            rows.append(record)
    if not rows:
        raise ValueError(f"No epoch history: {source}")
    epochs = [r["epoch"] for r in rows]
    if epochs != list(range(epochs[0], epochs[-1]+1)):
        raise ValueError("Non-contiguous or duplicate epoch history")
    return rows


def official_history(path):
    if path.suffix == ".json":
        checkpoint = json.loads(path.read_text())
    else:
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    history = checkpoint["results"]
    epochs = history["n_updates"]
    fields = {"reconstruction_loss": "recon_loss", "embedding_loss": "embedding_loss", "total_loss": "train_loss"}
    if epochs != list(range(1, len(epochs)+1)) or any(len(history[key]) != len(epochs) for key in fields.values()):
        raise ValueError("Inconsistent official training history")
    # Do not fabricate epoch20 when the released save order kept only 1..19.
    return [{"epoch": epoch, **{key: history[source][i] for key, source in fields.items()}}
            for i, epoch in enumerate(epochs)]


def plot_histories(histories, output):
    fields = ("epoch", "reconstruction_loss", "embedding_loss", "total_loss", "validation_loss", "learning_rate")
    records = [{"model": name, **{key: row.get(key) for key in fields}}
               for name, rows in histories.items() for row in rows]
    write_csv(output / "loss_by_epoch.csv", records)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, key, title in zip(axes.flat, ("reconstruction_loss", "total_loss", "embedding_loss", "validation_loss"),
                              ("Training reconstruction MSE", "Training total loss", "Weighted VQ loss (log scale)",
                               "Periodic validation total (first batch ONLY)")):
        for name, rows in histories.items():
            selected = [row for row in rows if row.get(key) is not None]
            if selected:
                ax.plot([row["epoch"] for row in selected], [row[key] for row in selected], label=name)
        if key == "embedding_loss":
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle("Observed epoch histories: unsmoothed, no padding | different training data/settings are not matched evaluations", fontsize=10)
    save_figure(fig, output / "loss_curves")
    (output / "loss_protocol.json").write_text(json.dumps({
        "validation": "Last periodic first-global-batch total loss, NOT full held-out evaluation; dropout differs from training.",
        "official": "Use only observed history; released epoch20 checkpoint may contain only epochs1..19.",
        "epochs": {name: [r["epoch"] for r in rows] for name, rows in histories.items()},
        "smoothed": False,
    }, indent=2)+"\n")


def parse_named(values):
    result = {}
    for item in values:
        name, separator, value = item.partition("=")
        if not separator or not name or name in result:
            raise ValueError("Expected unique LABEL=PATH arguments")
        result[name] = Path(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", default=[], help="LABEL=result.json; first is baseline")
    parser.add_argument("--training", action="append", default=[], help="LABEL=training directory")
    parser.add_argument("--official-history", type=Path, help="Released .pth or extracted history JSON")
    parser.add_argument("--paper-reference", type=Path, help="Optional local Table10 calibration JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not (args.evaluation or args.training or args.official_history):
        parser.error("Provide evaluations or training histories")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.evaluation:
        paper = None
        if args.paper_reference:
            raw = json.loads(args.paper_reference.read_text())
            paper = raw.get("paper_table_10", raw)
        compare(parse_named(args.evaluation), args.output_dir, paper)
    histories = {name: load_history(path) for name, path in parse_named(args.training).items()}
    if args.official_history:
        histories["Official"] = official_history(args.official_history)
    if histories:
        plot_histories(histories, args.output_dir)


if __name__ == "__main__":
    main()
