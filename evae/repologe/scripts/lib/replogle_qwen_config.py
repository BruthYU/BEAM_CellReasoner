"""Standard-library-only Qwen planning; model/data checks run before submission."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ART = ROOT / "artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
AB_CONFIG = ROOT / "config/local/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-v1"
BASE_CONFIG = ROOT / "config/local/replogle_llm_control/qwen35-08b-fsq-nb-original-split-control1-v1/train.json"
C_CONFIG = ROOT / "config/local/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-c-v1/C.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def runtime_config(cfg, *, gpus=8, batch_size=16, global_batch=128, cpus=48,
                   memory_gib=195, node=None, charged_group="beam_gpu", phase="formal",
                   gradient_checkpointing=True):
    if gpus not in (1, 2, 4, 8) or min(batch_size, global_batch, cpus, memory_gib) < 1:
        raise ValueError("Positive resources and 1/2/4/8 GPUs required")
    if cpus < gpus or global_batch % (gpus * batch_size):
        raise ValueError("CPU >= GPUs; global batch must be divisible by GPUs * per-device batch")
    cfg = {**cfg, "gpus": gpus, "per_device_batch_size": batch_size,
           "global_batch_size": global_batch,
           "gradient_accumulation_steps": global_batch // (gpus * batch_size),
           "cpu_threads_per_process": cpus // gpus, "total_cpu_threads": cpus,
           "memory_gib": memory_gib, "node": node, "charged_group": charged_group,
           "gradient_checkpointing": gradient_checkpointing, "phase": phase}
    if phase == "smoke":
        cfg.update(max_steps=10, train_cell_limit=2048, eval_steps=5, save_steps=5,
                   logging_steps=1, save_only_model=True)
    elif phase == "benchmark":
        cfg.update(max_steps=200, logging_steps=20)
    elif phase != "formal":
        raise ValueError(phase)
    return cfg


PRESETS = {
    "original-text": (ROOT / "config/local/replogle_llm/qwen35-08b-ar-fsq-nb-v1/formal-template.json", 0),
    "original-control1": (BASE_CONFIG, 1),
    "hepg2-a": (AB_CONFIG / "A.json", 0),
    "hepg2-b": (AB_CONFIG / "B.json", 1),
    "hepg2-c": (C_CONFIG, 8),
}
OUTPUT_ROOT = ROOT / "outputs/replogle_qwen"
REQUEST_ROOT = ROOT / "artifacts/replogle_qwen"


def preset_config(preset, *, control_cells=None, **resources):
    path, default_n = PRESETS[preset]
    n = default_n if control_cells is None else control_cells
    if preset == "hepg2-c":
        if n < 1:
            raise ValueError("C requires a positive --control-cells-nums")
    elif n != default_n:
        raise ValueError(f"{preset} requires {default_n} control cells; use hepg2-c for configurable N")
    cfg = runtime_config(read_json(path), **resources)
    cfg.update(preset=preset, control_cells=n, output_root=str((OUTPUT_ROOT / preset).relative_to(ROOT)),
               max_sequence_length=256 + 66 * n if preset == "hepg2-c" else 256)
    cfg.pop("output_dir", None)
    cfg["source_config"] = str(path.relative_to(ROOT))
    return cfg



def check_assets(cfg):
    """Early actionable presence/context check; frozen hashes are checked separately."""
    art = ROOT / cfg["artifacts"]
    initial = DEFAULT_ART / "initial_text_model"
    needed = [initial / name for name in
              ("config.json", "model.safetensors", "tokenizer.json", "export.json")]
    needed += [DEFAULT_ART / "cache" / name for name in
               ("token_ids.npy", "condition_rows.npy", "completion.json")]
    split_art = ROOT / cfg.get("split_artifacts", cfg["artifacts"])
    needed += [split_art / "sources.json", split_art / "cell_fold.npy"]
    if split_art != art:
        needed.append(split_art / "ready_inputs.json")
    needed += ([art / "cpu_preflight.json", art / "launch_sources.json"]
               if cfg["preset"] == "original-text" else [art / "ready_inputs.json"])
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing external Qwen assets (see SETUP.md):\n  " + "\n  ".join(missing))
    context = read_json(initial / "config.json")["max_position_embeddings"]
    if cfg["max_sequence_length"] > context:
        raise ValueError(f"Requested control cells exceed model context {context}")
