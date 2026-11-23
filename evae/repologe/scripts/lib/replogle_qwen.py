"""Qwen experiment presets: preserve datasets/prompts, share training/resources."""
# 共享预设与数据集选择；日常通过提交脚本选预设，无需修改本模块：
# bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-c --control-cells-nums 8 --dry-run

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap

from pathlib import Path
import numpy as np
from replogle_llm_common import ROOT, DEFAULT_ART, CellDataset, read_json, validation_indices
from replogle_control_common import ART as AB_ART, CellStore, ControlCellDataset
from replogle_control_baseline import ART as BASE_ART, BaselineControlDataset, runtime_config
from replogle_control_c import ControlSetDataset

from replogle_qwen_config import PRESETS, OUTPUT_ROOT, REQUEST_ROOT, preset_config, check_assets


def phase_config(cfg, phase):
    return runtime_config(cfg, **{k: cfg[v] for k, v in {
        "gpus": "gpus", "batch_size": "per_device_batch_size", "global_batch": "global_batch_size",
        "cpus": "total_cpu_threads", "memory_gib": "memory_gib", "node": "node",
        "charged_group": "charged_group", "gradient_checkpointing": "gradient_checkpointing"}.items()}, phase=phase)


def safe_output(path, cfg):
    path = Path(path).resolve()
    root = (OUTPUT_ROOT / cfg["preset"]).resolve()
    if cfg["preset"] == "original-control1":
        legacy = ROOT / "outputs/replogle_llm_control/qwen35-08b-fsq-nb-original-split-control1-v1"
        if path.is_relative_to(legacy):
            root = legacy
    assert path.is_relative_to(root) and path != root, f"Output outside isolated preset directory: {path}"
    return path


def verify_inputs(cfg):
    check_assets(cfg)
    preset = cfg["preset"]
    if preset == "original-text":
        art = ROOT / cfg.get("artifacts", str(DEFAULT_ART.relative_to(ROOT)))
        assert read_json(art / "cpu_preflight.json")["status"] == "passed"
        assert read_json(art / "cache/completion.json")["status"] == "complete"
        for name, expected in read_json(art / "launch_sources.json").items():
            _bootstrap.verify_input(name, expected)
    elif preset == "original-control1":
        from replogle_control_baseline import verify_inputs as verify
        verify()
    elif preset == "hepg2-c":
        from replogle_control_c import check_inputs
        check_inputs()
    else:
        from replogle_control_common import check_inputs
        check_inputs()
    # New requests additionally pin every current script, including shared modules.
    for name, expected in cfg.get("source_sha256", {}).items():
        assert _bootstrap.file_hash(ROOT / name) == expected, f"Submitted source changed: {name}"


def training_data(cfg):
    if cfg["preset"] == "original-text":
        art = ROOT / cfg.get("artifacts", str(DEFAULT_ART.relative_to(ROOT)))
        return art, np.load(art / "cell_fold.npy"), read_json(art / "sources.json"), validation_indices(art, cfg.get("validation_cells_per_condition", 20))
    art = BASE_ART if cfg["preset"] == "original-control1" else AB_ART
    store = CellStore(art)
    return store, store.fold, store.sources, np.load(art / "validation_monitor.npy")


class TextDataset(CellDataset):
    def inference_prefix(self, index, draw=0):
        return self.prefixes[int(self.condition_rows[int(self.indices[index])])]


def make_dataset(cfg, store, indices, tokenizer, training=False):
    preset = cfg["preset"]
    kwargs = dict(training=training, seed=cfg["seed"])
    if preset == "original-text":
        return TextDataset(store, indices, tokenizer)
    if preset == "original-control1":
        return BaselineControlDataset(store, indices, tokenizer, **kwargs)
    if preset == "hepg2-c":
        return ControlSetDataset(store, indices, tokenizer, cfg["control_cells"], **kwargs)
    return ControlCellDataset(store, indices, tokenizer, preset[-1].upper(), **kwargs)
