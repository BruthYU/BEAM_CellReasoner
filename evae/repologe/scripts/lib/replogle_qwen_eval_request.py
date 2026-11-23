"""Read-only model selection and isolated requests for standalone Qwen evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap
import json
import os

ROOT = _bootstrap.ROOT
OUTPUT_ROOT = ROOT / 'outputs/replogle_qwen_evaluation'
ART = ROOT / 'artifacts/replogle_qwen_evaluation'
PRESETS = ('original-text', 'original-control1')
ARTIFACTS = {
    'original-text': 'artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1',
    'original-control1': 'artifacts/replogle_llm_control/qwen35-08b-fsq-nb-original-split-control1-v1',
}
LEGACY_ROOTS = {
    'original-text': ROOT / 'outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1',
    'original-control1': ROOT / 'outputs/replogle_llm_control/qwen35-08b-fsq-nb-original-split-control1-v1',
}


def read(path):
    return json.loads(Path(path).read_text())


def source_config(preset, run_dir):
    """Accept either a run parent or its formal directory; never select a last-step model."""
    if preset not in PRESETS:
        raise ValueError('Unsupported evaluation preset')
    run = Path(run_dir).resolve()
    formal = run / 'formal' if (run / 'formal/completion.json').is_file() else run
    roots = (LEGACY_ROOTS[preset].resolve(), (ROOT / 'outputs/replogle_qwen' / preset).resolve())
    if not any(formal.is_relative_to(p) and formal != p for p in roots):
        raise ValueError('Training directory does not belong to the selected preset')
    if formal.name != 'formal':
        raise ValueError('Use the completed formal training directory, not smoke/benchmark')
    cfg = read(formal / 'run_config.json')
    done = read(formal / 'completion.json')
    if done.get('status') != 'complete' or done.get('phase', 'formal') != 'formal':
        raise ValueError('A completed formal training run is required')
    if cfg.get('preset', preset) != preset or cfg.get('artifacts') != ARTIFACTS[preset]:
        raise ValueError('Training configuration does not match the requested preset/split')
    n = 0 if preset == 'original-text' else 1
    if cfg.get('control_cells', n) != n:
        raise ValueError('Wrong control-cell count in training configuration')
    if not done.get('best_checkpoint'):
        raise ValueError('Training did not record a validation-selected checkpoint')
    selected = Path(done['best_checkpoint']).resolve()
    if not selected.is_relative_to(formal):
        raise ValueError('Selected checkpoint is outside its training run')
    for directory in (selected, formal / 'final'):
        if not (directory / 'model.safetensors').is_file():
            raise ValueError(f'Missing selected model: {directory}')
    return dict(mode='standalone_evaluation', preset=preset, artifacts=ARTIFACTS[preset],
        control_cells=n, seed=cfg.get('seed', 420), source_formal_dir=str(formal),
        selected_checkpoint=str(selected), source_metadata_sha256={
            str(formal / name): _bootstrap.file_hash(formal / name)
            for name in ('run_config.json', 'completion.json')})


def safe_output(path, cfg):
    path = Path(path).resolve()
    root = (OUTPUT_ROOT / cfg['preset']).resolve()
    if cfg.get('mode') != 'standalone_evaluation' or path.parent != root:
        raise ValueError('Standalone evaluation must use its isolated preset/run-id directory')
    return path


def verify_source(cfg, check_weights=False):
    current = source_config(cfg['preset'], cfg['source_formal_dir'])
    for key in ('source_formal_dir', 'selected_checkpoint', 'source_metadata_sha256'):
        if cfg[key] != current[key]:
            raise ValueError(f'Training source changed: {key}')
    if check_weights:
        final_hash = _bootstrap.file_hash(Path(cfg['source_formal_dir']) / 'final/model.safetensors')
        if final_hash != _bootstrap.file_hash(Path(cfg['selected_checkpoint']) / 'model.safetensors'):
            raise ValueError('Final weights differ from the validation-selected checkpoint')
        if cfg.get('source_model_sha256', final_hash) != final_hash:
            raise ValueError('Submitted model weights changed')
        return final_hash


def initialize(request_path):
    """Worker-only materialization; no writes to the source training directory."""
    from replogle_qwen import verify_inputs
    cfg = read(request_path)
    verify_source(cfg, check_weights=True)
    verify_inputs(cfg)
    if os.environ.get('KUBEBRAIN_QUOTA_GROUP') != cfg['charged_group']:
        raise ValueError('Worker quota group differs from its request')
    if cfg.get('node') and os.environ.get('KUBEBRAIN_NODE_NAME') != cfg['node']:
        raise ValueError('Worker node differs from its request')
    run = safe_output(cfg['output_dir'], cfg)
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir()  # Exclusive run ID: never reuse/overwrite a previous evaluation.
    (run / 'request.json').write_text(json.dumps(cfg, indent=2) + '\n')
    return run
