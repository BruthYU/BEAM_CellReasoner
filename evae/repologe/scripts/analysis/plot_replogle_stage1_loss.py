"""Export Stage 1 or AR Stage 2 loss curves from the saved training log."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="artifacts/replogle_stage1/formal-training.log")
    parser.add_argument("--run-dir", default="outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401")
    parser.add_argument("--smooth-steps", type=int, default=500)
    parser.add_argument("--output-dir", help="Override the run's analysis/loss_curve directory")
    parser.add_argument("--wait-for-training", action="store_true",
                        help="Wait for the final checkpoint and completed training log before plotting")
    parser.add_argument("--wait-timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    assert args.smooth_steps > 0
    source = ROOT / args.log
    run_dir = ROOT / args.run_dir
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    stage = 2 if config["model"]["type"] == "autoregressive" else 1
    vae_config = (yaml.safe_load((ROOT / config["model"]["vae_path"] / "config.yaml").read_text())
                  if stage == 2 else config)
    head = vae_config["model"]["output_head"]["type"]
    assert head in {"mse", "nb"}, head
    metric = "token_cross_entropy" if stage == 2 else ("nb_nll" if head == "nb" else "mse")
    metric_label = "token CE" if stage == 2 else ("NB NLL" if head == "nb" else "MSE")
    final_step = config["n_iters"]
    eval_freq = config["eval_freq"]
    warmup = config["warmup_steps"]
    if args.wait_for_training:
        deadline = time.monotonic() + args.wait_timeout_seconds
        print(f"Waiting for {run_dir}, final step {final_step}", flush=True)
        while True:
            pointer = run_dir / "checkpoints/latest"
            if pointer.exists() and pointer.read_text().strip() == f"step_{final_step:07d}_final":
                current = source.read_text(errors="replace")
                if f"step={final_step:07d}  val_loss=" in current and "Training complete." in current:
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Training did not finish before plotting deadline: {run_dir}")
            time.sleep(30)
    text = source.read_text(errors="replace")
    # tqdm often repeats a step; keep its last printed batch loss.
    train = {int(step): float(loss) for loss, step in re.findall(
        r"loss=([\d.]+), lr=[^,\]]+, step=(\d+)", text)}
    val = {int(step): float(loss) for step, loss in re.findall(
        r"step=(\d+)  val_loss=([\d.]+)", text)}
    assert train and val
    steps = np.array(sorted(train), dtype=np.int64)
    losses = np.array([train[int(step)] for step in steps])
    val_steps = np.array(sorted(val), dtype=np.int64)
    val_losses = np.array([val[int(step)] for step in val_steps])
    assert steps[0] == 0 and steps[-1] == final_step
    np.testing.assert_array_equal(val_steps, np.arange(eval_freq, final_step + 1, eval_freq))
    assert np.isfinite(losses).all() and np.isfinite(val_losses).all()
    # The window covers optimizer step numbers, ignoring unprinted steps.
    # No raw training losses are interpolated or filled.
    starts = np.searchsorted(steps, steps - args.smooth_steps + 1)
    cumulative = np.concatenate([[0.0], losses.cumsum()])
    smooth = (cumulative[1:] - cumulative[starts]) / (np.arange(len(steps)) + 1 - starts)
    output = ROOT / args.output_dir if args.output_dir else run_dir / "analysis/loss_curve"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "train_loss.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", f"train_batch_{metric}",
                         f"train_moving_mean_{args.smooth_steps}_steps"])
        writer.writerows(zip(steps.tolist(), losses.tolist(), smooth.tolist()))
    with (output / "validation_loss.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", f"heldout_{metric}" if stage == 2 else f"heldout_reconstruction_{metric}"])
        writer.writerows(zip(val_steps.tolist(), val_losses.tolist()))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), layout="constrained")
    blue, orange = "#2563a6", "#cf6422"
    for ax in axes:
        ax.plot(steps, losses, color=blue, alpha=.16, lw=.45,
                label="Train batch", rasterized=True)
        ax.plot(steps, smooth, color=blue, lw=1.8,
                label=f"Train ({args.smooth_steps}-step moving mean)")
        ax.plot(val_steps, val_losses, color=orange, marker="o", ms=4,
                lw=1.7, label="Held-out HepG2 (372 targets)")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Token cross-entropy (nats / token)" if stage == 2 else
                      ("Reconstruction NB NLL (sum over 2,000 genes / cell)" if head == "nb"
                       else "Reconstruction MSE (log1p expression)"))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_title("Full training run", loc="left", fontweight="bold")
    xmax = final_step * 1.0266666667
    axes[0].set_xlim(0, xmax)
    full_lower = min(losses.min(), val_losses.min())
    full_upper = max(losses.max(), val_losses.max())
    full_padding = (full_upper - full_lower) * .05
    axes[0].set_ylim(full_lower - full_padding, full_upper + full_padding)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set_title(f"After warmup (step {warmup:,} onward)", loc="left", fontweight="bold")
    late = losses[steps >= warmup]
    lower = min(late.min(), val_losses.min())
    upper = max(late.max(), val_losses.max())
    padding = (upper - lower) * .15
    axes[1].set_xlim(warmup, xmax)
    axes[1].set_ylim(lower - padding, upper + padding)
    axes[1].annotate(f"Final held-out {metric_label}: {val_losses[-1]:.4f}",
                     xy=(val_steps[-1], val_losses[-1]), xytext=(final_step * 36500 / 75000, upper),
                     color=orange, fontsize=10,
                     arrowprops={"arrowstyle": "->", "color": orange, "lw": 1})
    model = vae_config["model"]
    title = f"AR–FSQ–{head.upper()}" if stage == 2 else f"FSQ{'–NB' if head == 'nb' else ''} tokenizer"
    fig.suptitle(f"Replogle Stage {stage} · {title}\n"
                 f"{model['num_genes']:,} genes · {model['num_latent_tokens']} tokens/cell · "
                 f"{model['quantizer']['codebook_size']} codes/token · seed {config['seed']}",
                 fontsize=14, fontweight="bold")
    fig.savefig(output / f"stage{stage}_loss_curve.png", dpi=240)
    fig.savefig(output / f"stage{stage}_loss_curve.pdf", dpi=240)
    plt.close(fig)
    metadata = {
        "source_log": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "training_points": len(steps), "validation_points": len(val_steps),
        "unprinted_training_steps": int(steps[-1] + 1 - len(steps)),
        "source_precision_decimal_places": 4,
        "training_loss_definition": "Loss of the printed training minibatch, not an epoch mean",
        "validation_loss_definition": f"Original trainer's full held-out validation loss every {eval_freq:,} steps",
        "smoothing": f"Trailing {args.smooth_steps} optimizer steps, observed losses only; no interpolation",
        f"final_heldout_{metric}": float(val_losses[-1]),
        "stage": stage,
        "head": head,
        "loss_definition": "Teacher-forced target-token cross entropy, averaged over cells and token positions"
            if stage == 2 else ("Raw-count negative binomial NLL, summed over genes and averaged over cells"
            if head == "nb" else "Mean squared error averaged over cells and genes, normalized log1p expression"),
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
