"""Immutable C evaluation snapshot and references to the frozen A/B split."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap

from pathlib import Path
from replogle_llm_common import ROOT,read_json,sha
from replogle_control_c import ART,check_inputs
from replogle_control_eval_common import verify_inputs as verify_ab
EVAL_ART=ART/'evaluation_observed_depth_ckpt11000_n8_v1'
EVAL_ROOT=ROOT/'outputs/replogle_llm_control_evaluation/c-observed-depth-ckpt11000-n8-v1'
FILES = ['replogle_control_c_eval_common.py', 'prepare_replogle_control_c_eval.py', 'eval_replogle_qwen.py', 'replogle_qwen_legacy_eval.py', 'replogle_qwen_inference.py', 'replogle_qwen_metrics.py', 'replogle_qwen.py', 'run_replogle_qwen_eval.sh', 'submit_replogle_control_eval.sh', 'submit_replogle_control_eval.py', 'run_replogle_control_eval_worker.sh']
def verify_inputs():
    check_inputs();verify_ab()
    manifest=read_json(EVAL_ART/'ready_inputs.json')
    assert manifest['status']=='cpu_ready'
    for name,expected in manifest['sha256'].items():_bootstrap.verify_input(name, expected)
