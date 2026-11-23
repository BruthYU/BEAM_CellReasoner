"""Plot formal Qwen SFT loss, preserving the pre-restore validation trajectory."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(
        "outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1/formal"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    source = run / "metrics.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    cfg = json.loads((run / "run_config.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    state = json.loads((run / "trainer_state.json").read_text())
    train = [r for r in rows if "loss" in r]
    # Trainer logs another evaluate() after loading the best checkpoint. Keep
    # the FIRST evaluation at each step for the actual chronological curve.
    val = {}
    restored = []
    for row in rows:
        if "eval_loss" in row:
            if row["step"] in val:
                restored.append(row)
            else:
                val[row["step"]] = row
    steps = np.array([r["step"] for r in train])
    losses = np.array([r["loss"] for r in train])
    vsteps = np.array(sorted(val))
    vlosses = np.array([val[int(s)]["eval_loss"] for s in vsteps])
    assert np.all(np.diff(steps) > 0) and np.isfinite(losses).all()
    assert np.isfinite(vlosses).all() and vsteps[-1] == completion["steps"]
    best_step = int(Path(completion["best_checkpoint"]).name.split("-")[-1])
    best_loss = val[best_step]["eval_loss"]
    assert math.isclose(best_loss, state["best_metric"], abs_tol=1e-7)
    assert len(restored) == 1 and restored[0]["step"] == completion["steps"]
    assert math.isclose(restored[0]["eval_loss"], best_loss, abs_tol=1e-7)
    warmup = math.ceil(completion["steps"] * cfg["warmup_ratio"])
    # Each input loss already averages its 20-step Trainer logging interval.
    # Apply an additional trailing mean over 200 optimizer steps for display.
    starts = np.searchsorted(steps, steps - 200 + 1)
    cumulative = np.r_[0., losses.cumsum()]
    smooth = (cumulative[1:] - cumulative[starts]) / (np.arange(len(steps)) + 1 - starts)
    output = run / "analysis/loss_curve"
    output.mkdir(parents=True, exist_ok=True)
    for name, header, data in [
        ("train_loss.csv", ["step", "train_logged_mean_ce", "trailing_200_step_mean_ce"],
         zip(steps, losses, smooth)),
        ("validation_loss.csv", ["step", "internal_validation_ce_before_best_restore"],
         zip(vsteps, vlosses)),
    ]:
        with (output / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(data)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8))
    fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.78, wspace=.18)
    blue, orange = "#2563a6", "#cf6422"
    for ax in axes:
        ax.plot(steps, losses, color=blue, lw=.65, alpha=.26,
                label="Train (logged 20-step mean)")
        ax.plot(steps, smooth, color=blue, lw=1.8, label="Train (200-step trailing mean)")
        ax.plot(vsteps, vlosses, color=orange, lw=1.6, marker="o", ms=3.3,
                label="Internal validation (4,000 cells)")
        ax.axvline(warmup, color="#7c8798", lw=1, ls=":")
        ax.scatter([best_step], [best_loss], color=orange, marker="*", s=150, zorder=5)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Completion token cross-entropy (nats / token)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_title("Full formal training run", loc="left", fontweight="bold")
    axes[0].set_xlim(0, completion["steps"] * 1.025)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].text(.98, .57, f"Warmup: {warmup:,} steps\n"
                 f"Selected checkpoint: {best_step:,}\nBest validation CE: {best_loss:.4f}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=10)
    axes[1].set_title(f"After warmup (step {warmup:,} onward)", loc="left", fontweight="bold")
    late = np.r_[losses[steps >= warmup], vlosses[vsteps >= warmup]]
    padding = np.ptp(late) * .12
    axes[1].set_xlim(warmup, completion["steps"] * 1.025)
    axes[1].set_ylim(late.min() - padding, late.max() + padding)
    axes[1].annotate(f"Best: {best_loss:.4f} at step {best_step:,}",
                     xy=(best_step, best_loss), xytext=(9000, 1.53), color=orange,
                     arrowprops={"arrowstyle": "->", "color": orange}, fontsize=10)
    axes[1].annotate(f"Last-step validation: {vlosses[-1]:.4f}",
                     xy=(vsteps[-1], vlosses[-1]), xytext=(12200, 1.62), color=orange,
                     arrowprops={"arrowstyle": "->", "color": orange}, fontsize=10)
    fig.suptitle("Replogle Stage 2 · Qwen3.5-0.8B + FSQ-NB\n"
                 "Full SFT · 2 H200 GPUs · global batch 128 · 3 epochs · learning rate 3e-5",
                 fontweight="bold", fontsize=14, y=.975)
    fig.text(.5, .025, "Validation: fixed 4,000-cell monitor from 200 held-out training conditions; "
             "official 372-target test set excluded.\n"
             "Final model restores step 14,000; the post-restore evaluation is not plotted at step 21,669.",
             ha="center", va="bottom", fontsize=9, color="#4b5563")
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"qwen_stage2_loss_curve.{ext}", dpi=220)
    plt.close(fig)
    provenance = {
        "source_metrics": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "training_points": len(train), "validation_points": len(val),
        "warmup_steps": warmup, "best_step": best_step, "best_validation_loss": best_loss,
        "last_training_step_validation_loss": float(vlosses[-1]),
        "post_best_restore_evaluation": restored,
        "smoothing": "Trailing 200 optimizer steps of logged interval-mean training CE",
        "validation": "Fixed monitor of 20 cells per condition, 200 internal validation conditions",
        "objective": "Completion-only next-token CE (64 FSQ codes + cell_end + EOS); prefix masked",
        "official_test_used": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), **provenance}, indent=2))


if __name__ == "__main__":
    main()
