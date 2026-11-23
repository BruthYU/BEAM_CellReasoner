"""Plot A/B formal loss without conflating observed minima and saved checkpoints."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


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

ROOT = Path(__file__).resolve().parents[2]
NAME = "qwen35-08b-fsq-nb-hepg2-v1"
ART = ROOT / "artifacts/replogle_llm_control" / NAME
OUT = ART / "analysis/loss_curve"


def load_run(record):
    run = ROOT / "outputs/replogle_llm_control" / NAME / record["variant"] / record["run_id"] / "formal"
    rows = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    cfg = json.loads((run / "run_config.json").read_text())
    done = json.loads((run / "completion.json").read_text())
    assert done["status"] == "complete" and done["steps"] == 17418
    train = [r for r in rows if "loss" in r]
    validation, restored = {}, []
    for r in rows:
        if "eval_loss" in r:
            if r["step"] in validation:
                restored.append(r)
            else:
                validation[r["step"]] = r
    assert len(restored) == 1 and restored[0]["step"] == done["steps"]
    selected = int(Path(done["best_checkpoint"]).name.split("-")[-1])
    exported_loss = done["validation_metrics"]["eval_loss"]
    assert math.isclose(exported_loss, validation[selected]["eval_loss"], abs_tol=1e-7)
    assert math.isclose(exported_loss, restored[0]["eval_loss"], abs_tol=1e-7)
    best = min(validation.values(), key=lambda r: r["eval_loss"])
    steps = np.array([r["step"] for r in train])
    losses = np.array([r["loss"] for r in train])
    starts = np.searchsorted(steps, steps - 199)
    cumulative = np.r_[0., losses.cumsum()]
    smooth = (cumulative[1:] - cumulative[starts]) / (np.arange(len(steps)) + 1 - starts)
    assert np.all(np.diff(steps) > 0) and np.isfinite(losses).all()
    vsteps = np.array(sorted(validation))
    vlosses = np.array([validation[int(s)]["eval_loss"] for s in vsteps])
    summary = {"variant": record["variant"], "run": str(run.relative_to(ROOT)),
        "steps": done["steps"], "gpus": done["gpus"], "validation_cells": cfg["validation_monitor_cells"],
        "train_runtime_seconds": done["train_metrics"]["train_runtime"],
        "last_logged_train_loss": float(losses[-1]), "last_logged_train_step": int(steps[-1]),
        "observed_best_step": best["step"], "observed_best_validation_loss": best["eval_loss"],
        "exported_checkpoint_step": selected, "exported_model_validation_loss": exported_loss,
        "last_step_validation_before_restore": float(vlosses[-1]),
        "test_evaluation_performed": done["test_evaluation_performed"],
        "metrics_sha256": hashlib.sha256((run / "metrics.jsonl").read_bytes()).hexdigest(),
        "note": "Validation every 500 steps, saves every 1000; step-11500 minimum was not saved. Export restores step 11000."}
    for suffix, headers, data in [
        ("train", ["step", "logged_train_ce", "trailing_200_step_mean_ce"], zip(steps, losses, smooth)),
        ("validation", ["step", "validation_ce_before_restore"], zip(vsteps, vlosses)),
    ]:
        with (OUT / f"{record['variant']}_{suffix}.csv").open("w", newline="") as f:
            writer = csv.writer(f); writer.writerow(headers); writer.writerows(data)
    return summary, steps, losses, smooth, vsteps, vlosses


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((ART / "active_submissions.json").read_text())
    runs = [load_run(r) for r in records]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    fig.subplots_adjust(left=.065, right=.985, bottom=.22, top=.82, wspace=.17)
    for (summary, steps, losses, smooth, vsteps, vlosses), color in zip(runs, ["#2563a6", "#cf6422"]):
        label = "A: text only" if summary["variant"] == "A" else "B: text + 1 control"
        for ax in axes:
            ax.plot(steps, losses, color=color, lw=.5, alpha=.2)
            ax.plot(steps, smooth, color=color, lw=1.7, label=f"{label} / train")
            ax.plot(vsteps, vlosses, color=color, lw=1.7, ls="--", marker="o", ms=2.5,
                    label=f"{label} / validation")
            ax.scatter([summary["observed_best_step"]], [summary["observed_best_validation_loss"]],
                       marker="*", s=100, color=color, zorder=5)
            ax.scatter([summary["exported_checkpoint_step"]], [summary["exported_model_validation_loss"]],
                       marker="D", s=45, facecolors="white", edgecolors=color, zorder=5)
        axes[1].annotate(f"{summary['variant']} saved model: {summary['exported_model_validation_loss']:.4f}",
            xy=(summary["exported_checkpoint_step"], summary["exported_model_validation_loss"]),
            xytext=(12500, 1.55 if summary["variant"] == "A" else 1.48), color=color,
            fontsize=9, arrowprops={"arrowstyle": "->", "color": color})
    axes[0].set_title("Full formal training", loc="left", fontweight="bold")
    axes[0].set_xlim(0, 17800)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].axvline(math.ceil(17418 * .03), color="gray", lw=1, ls=":")
    axes[1].set_title("Zoom: step 1,000 onward", loc="left", fontweight="bold")
    axes[1].set_xlim(1000, 17800)
    axes[1].set_ylim(1.26, 1.68)
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Completion token cross-entropy (nats/token)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Replogle · Qwen3.5-0.8B + FSQ-NB · A/B control-input experiments\n"
                 "4 H200 GPUs each · global batch 128 · 3 epochs · 17,418 updates", fontsize=14, fontweight="bold")
    fig.text(.5, .045, "Validation: fixed 6,360-cell monitor from 318 held-out source-line conditions; HepG2 excluded.\n"
             "Train: 200-step trailing mean. Star: observed minimum (11,500, not saved). Diamond: exported checkpoint (11,000).\n"
             "The restored-model evaluation is marked at 11,000, not appended to the chronological curve at 17,418.",
             ha="center", fontsize=9, color="#4b5563")
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"replogle_control_ab_loss.{ext}", dpi=200)
    plt.close(fig)
    (OUT / "summary.json").write_text(json.dumps([r[0] for r in runs], indent=2) + "\n")
    print(json.dumps([r[0] for r in runs], indent=2))


if __name__ == "__main__":
    main()
