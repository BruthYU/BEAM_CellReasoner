"""Gate dual-GPU full SFT with smoke checks and internal-validation selection."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import datetime
import fcntl
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

from replogle_llm_common import ROOT, DEFAULT_ART, read_json, save_json, sha


def main():
    art = DEFAULT_ART
    lock = (art / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = read_json(art / "pipeline_request.json")
    for name, expected in read_json(art / "launch_sources.json").items():
        _bootstrap.verify_input(name, expected)
    assert read_json(art / "cache/completion.json")["status"] == "complete"
    assert read_json(art / "cpu_preflight.json")["status"] == "passed"
    env = os.environ.copy()

    def phase(name, command):
        save_json(art / "worker_phase.json", {"phase": name, "command": command,
            "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        print(name, shlex.join(command), flush=True)
        with (art / f"{name}.log").open("ab", buffering=0) as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while chunk := proc.stdout.read1(64 * 1024):
                log.write(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            code = proc.wait()
        if code:
            raise RuntimeError(f"{name} failed with exit {code}; formal training is gated on successful checks")

    def train(name, config):
        phase(name, ["/opt/sc-evae/bin/torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=2",
                     "scripts/training/train_replogle_llm.py", "--config", config])

    train("smoke-training", request["smoke_config"])
    smoke = read_json(ROOT / request["smoke_config"])
    smoke_run = ROOT / smoke["output_dir"]
    phase("smoke-generation", [sys.executable, "scripts/checks/check_replogle_llm_generation.py",
          "--model-dir", str(smoke_run / "final"), "--output", str(smoke_run / "generation_acceptance.json")])
    assert read_json(smoke_run / "generation_acceptance.json")["status"] == "passed"
    candidates = []
    for config in request["pilot_configs"]:
        cfg = read_json(ROOT / config)
        run = ROOT / cfg["output_dir"]
        name = run.name
        train(name + "-training", config)
        phase(name + "-validation", [sys.executable, "scripts/evaluation/eval_replogle_qwen.py", "--protocol", "legacy-text",
              "--model-dir", str(run / "final"), "--output-dir", str(run / "quick_validation"), "--split", "quick"])
        metrics = read_json(run / "quick_validation/summary_metrics_symbol_aligned.json")
        score = metrics["cell_eval"]["pearson_delta"]["mean"]
        assert math.isfinite(score)
        candidates.append({"config": config, "learning_rate": cfg["learning_rate"],
                           "quick_validation_pearson_delta": score,
                           "validation_token_loss": read_json(run / "completion.json")["validation_metrics"]["eval_loss"]})
    best = max(candidates, key=lambda row: (row["quick_validation_pearson_delta"], -row["validation_token_loss"]))
    save_json(art / "selection.json", {"candidates": candidates, "selected": best,
        "criterion": "Quick internal-validation population Pearson delta; token loss breaks ties",
        "test_set_used": False})
    formal = read_json(ROOT / request["formal_template"])
    formal["learning_rate"] = best["learning_rate"]
    selected_config = art / "formal_selected.json"
    save_json(selected_config, formal)
    train("formal-training", str(selected_config.relative_to(ROOT)))
    final = ROOT / formal["output_dir"] / "final"
    for split in ["validation", "test"]:
        phase(split + "-evaluation", [sys.executable, "scripts/evaluation/eval_replogle_qwen.py", "--protocol", "legacy-text",
              "--model-dir", str(final), "--output-dir", str(ROOT / formal["output_dir"] / (split + "_evaluation")),
              "--split", split])
    save_json(art / "completion.json", {"status": "complete", "formal_run": formal["output_dir"],
        "selection": best, "validation": formal["output_dir"] + "/validation_evaluation/summary_metrics_symbol_aligned.json",
        "test": formal["output_dir"] + "/test_evaluation/summary_metrics_symbol_aligned.json",
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    save_json(art / "worker_phase.json", {"phase": "complete"})


if __name__ == "__main__":
    main()
