"""Original-split Qwen full SFT with one unpaired control, isolated outputs."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
from pathlib import Path
from replogle_llm_common import read_json
from replogle_control_baseline import runtime_config, safe_output, verify_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--phase", choices=["smoke", "benchmark", "formal"], required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    requested = read_json(args.config)
    cfg = runtime_config(requested, **{k: requested[v] for k, v in {
        "gpus": "gpus", "batch_size": "per_device_batch_size", "global_batch": "global_batch_size",
        "cpus": "total_cpu_threads", "memory_gib": "memory_gib", "node": "node",
        "charged_group": "charged_group", "gradient_checkpointing": "gradient_checkpointing"}.items()}, phase=args.phase)
    output = safe_output(args.output)
    cfg.update(preset='original-control1', phase=args.phase)
    from train_replogle_qwen import train
    train(cfg, output, verify=verify_inputs)


if __name__ == "__main__":
    main()
