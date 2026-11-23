"""Compatibility CLI for the original text-only Qwen experiment."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap

import argparse
from pathlib import Path
from replogle_llm_common import ROOT, DEFAULT_ART, read_json
from replogle_qwen_callbacks import Progress
from train_replogle_qwen import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    cfg = read_json(args.config)
    cfg.update(preset="original-text", control_cells=0)
    cfg.setdefault("max_sequence_length", 256)
    output = (ROOT / cfg["output_dir"]).resolve()
    allowed = (ROOT / "outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1").resolve()
    assert output.is_relative_to(allowed) and output != allowed
    train(cfg, output, resume=args.resume)


if __name__ == "__main__":
    main()
