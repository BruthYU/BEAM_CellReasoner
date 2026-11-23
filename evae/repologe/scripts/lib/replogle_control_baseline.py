"""One control cell on the exact original Qwen split; old experiments are read-only."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


from pathlib import Path

from replogle_llm_common import ROOT, DEFAULT_ART, read_json, sha, training_arguments
from replogle_control_common import ControlCellDataset
from replogle_qwen_config import runtime_config

NAME = "qwen35-08b-fsq-nb-original-split-control1-v1"
ART = ROOT / "artifacts/replogle_llm_control" / NAME
CONFIG = ROOT / "config/local/replogle_llm_control" / NAME / "train.json"
OUTPUT_ROOT = ROOT / "outputs/replogle_llm_control" / NAME
FILES = ['replogle_control_baseline.py', 'prepare_replogle_control_baseline.py', 'train_replogle_control_baseline.py', 'run_replogle_control_baseline_worker.sh', 'replogle_control_common.py', 'replogle_llm_common.py', 'train_replogle_llm.py', 'train_replogle_control_ab.py', 'check_replogle_control_baseline_eval.py', 'replogle_control_eval_common.py', 'replogle_control_c_eval_common.py', 'eval.py', 'eval_replogle_qwen.py', 'replogle_qwen_legacy_eval.py', 'replogle_qwen_inference.py', 'replogle_qwen_metrics.py', 'replogle_qwen.py', 'run_replogle_qwen_eval.sh', 'submit_replogle_control_eval.sh', 'submit_replogle_control_eval.py', 'run_replogle_control_eval_worker.sh', 'submit_replogle_qwen.sh', 'submit_replogle_qwen.py', 'run_replogle_qwen_worker.sh', 'train_replogle_qwen.py', 'replogle_qwen_callbacks.py']


def verify_inputs():
    manifest = read_json(ART / "ready_inputs.json")
    assert manifest["status"] == "cpu_ready"
    for name, expected in manifest["sha256"].items():
        _bootstrap.verify_input(name, expected)
    return manifest


def safe_output(path):
    path = Path(path).resolve()
    assert path.is_relative_to(OUTPUT_ROOT.resolve()) and path != OUTPUT_ROOT.resolve(), path
    return path



def make_training_arguments(cfg, output, cpu_check=False):
    args = training_arguments(cfg, output, cpu_check=cpu_check)
    # Installed Transformers resolves the deprecated alias only during __post_init__.
    # Set its canonical field too, otherwise --no-gradient-checkpointing is ignored.
    args.activation_checkpointing = args.gradient_checkpointing = cfg["gradient_checkpointing"]
    if cfg["phase"] == "benchmark":
        args.eval_strategy = args.save_strategy = "no"
        args.load_best_model_at_end = False
    return args


class BaselineControlDataset(ControlCellDataset):
    """Reuse B sampling/labels; append only Control Cell to the original prompt."""

    def __init__(self, store, indices, tokenizer, training=False, seed=420):
        super().__init__(store, indices, tokenizer, "B", training=training, seed=seed)
        marker = "__CONTROL_CODES_INSERTION__"
        for k in self.prefix_parts:
            text = store.conditions[k]["text"] + "\n\nControl Cell:\n" + marker
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            before, after = rendered.split(marker)
            self.prefix_parts[k] = (tokenizer.encode(before, add_special_tokens=False),
                                    tokenizer.encode(after, add_special_tokens=False))
        self.max_length = max(len(a) + len(b) + 66 + 67 for a, b in self.prefix_parts.values())
