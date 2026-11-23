"""Independent A/B full SFT using original Qwen weights and frozen cached NB codes."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
from pathlib import Path
from replogle_llm_common import ROOT, read_json
from replogle_control_common import CONFIG, runtime_config, safe_output, check_inputs
from replogle_qwen_callbacks import Throughput


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=["A", "B"])
    p.add_argument("--gpus", type=int, default=4, choices=[2, 4])
    p.add_argument("--phase", choices=["smoke", "benchmark", "formal"], default="formal")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    cfg = runtime_config(read_json(CONFIG / f"{args.variant}.json"), args.gpus, args.phase)
    output = safe_output(args.output)
    assert output.is_relative_to((ROOT / cfg["output_root"]).resolve())
    cfg.update(preset="hepg2-" + args.variant.lower(), phase=args.phase)
    from train_replogle_qwen import train
    train(cfg, output, verify=check_inputs, resume=args.resume)


if __name__ == "__main__":
    main()
