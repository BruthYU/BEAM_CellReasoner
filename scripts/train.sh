#!/usr/bin/env bash
# Active experiment: full-primary AdamW. Edit parameters here, not new launcher files.
GPUS="${GPUS:-4}"; CPUS="${CPUS:-32}"; MEMORY_MIB="${MEMORY_MIB:-128000}"
CHARGED_GROUP="${CHARGED_GROUP:-beam_gpu_pool}"
SESSION="${SESSION:-celltok-train}"
CONFIG="${CONFIG:-configs/tokenizer_stage1_full_primary.yaml}"
MANIFEST="${MANIFEST:-data/stage1/full_primary/fit_full_primary_20261026.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tokenizer/adamw_full_primary}"
OPTIMIZER="${OPTIMIZER:-adamw}"; LEARNING_RATE="${LEARNING_RATE:-0.0003}"
SCHEDULER="${SCHEDULER:-warmup_cosine}"; EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"; WORKERS="${WORKERS:-2}"
SAVE_EVERY="${SAVE_EVERY:-5}"; MAX_STEPS="${MAX_STEPS:-}"; RESUME="${RESUME:-}"
CACHE_DIR="${CACHE_DIR:-/tmp/celltok-training}"
# Historical 41.995M control (do not overwrite old output):
# MANIFEST=data/stage1/full_primary/fit_complete_datasets_20261025.csv; OUTPUT_DIR=outputs/tokenizer/adamw_control_v2
# SGD reproduction only; NOT scheduled for full-primary:
# OPTIMIZER=sgd; LEARNING_RATE=0.0001; SCHEDULER=reduce_on_plateau; EPOCHS=30; OUTPUT_DIR=outputs/tokenizer/sgd_control_v2
# Optional diagnostic, not a separate smoke launcher/config:
# EPOCHS=2; MAX_STEPS=102; SAVE_EVERY=1; OUTPUT_DIR=outputs/tokenizer/smoke
PARAM_NAMES=(CONFIG MANIFEST OUTPUT_DIR OPTIMIZER LEARNING_RATE SCHEDULER EPOCHS BATCH_SIZE WORKERS SAVE_EVERY MAX_STEPS RESUME CACHE_DIR)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
preflight() {
  if [[ -e "$OUTPUT_DIR/run_manifest.json" && -z "$RESUME" ]]; then
    echo "Existing run: $OUTPUT_DIR. Do not duplicate a live job; choose a NEW output or an explicit resume." >&2
    return 1
  fi
}
experiment() {
  if [[ "$MODE" != --dry-run ]]; then preflight; fi
  local -a extra=()
  [[ -z "$MAX_STEPS" ]] || extra+=(--max-train-steps "$MAX_STEPS")
  [[ -z "$RESUME" ]] || extra+=(--resume "$RESUME")
  run_python -m celltok train --config "$CONFIG" --manifest "$MANIFEST" \
    --num-gpus "$GPUS" --optimizer "$OPTIMIZER" --learning-rate "$LEARNING_RATE" \
    --scheduler "$SCHEDULER" --epochs "$EPOCHS" --global-batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" --local-cache-dir "$CACHE_DIR" --save-every-epochs "$SAVE_EVERY" \
    --output-dir "$OUTPUT_DIR" "${extra[@]}"
}
run_entry
