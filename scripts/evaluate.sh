#!/usr/bin/env bash
# Use identical CONFIG/MANIFEST/SPLIT/MAX_CELLS for every checkpoint comparison.
GPUS="${GPUS:-1}"; CPUS="${CPUS:-8}"; MEMORY_MIB="${MEMORY_MIB:-64000}"
CHARGED_GROUP="${CHARGED_GROUP:-beam_gpu_pool}"; SESSION="${SESSION:-celltok-evaluate}"
CONFIG="${CONFIG:-configs/tokenizer_stage1_full_primary.yaml}"
MANIFEST="${MANIFEST:-data/stage1/full_primary/fit_full_primary_20261026.csv}"
CHECKPOINT="${CHECKPOINT:-model/cell_tokenzier/model.pth}"; SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-outputs/evaluation/reconstruction.json}"
MAX_CELLS="${MAX_CELLS:-10000}"; ALL_CELLS="${ALL_CELLS:-0}"
DEVICE="${DEVICE:-cuda}"
# ALL_CELLS=1 streams the complete test split; does not accumulate full dense matrices.
# GPUS=0 DEVICE=cpu selects CPU evaluation.
PARAM_NAMES=(CONFIG MANIFEST CHECKPOINT SPLIT OUTPUT MAX_CELLS ALL_CELLS DEVICE)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
experiment() {
  local -a extra=()
  if [[ "$ALL_CELLS" == 1 ]]; then extra+=(--all-cells); else extra+=(--max-cells "$MAX_CELLS"); fi
  run_python -m celltok evaluate tokenizer --config "$CONFIG" --manifest "$MANIFEST" \
    --checkpoint "$CHECKPOINT" --split "$SPLIT" --device "$DEVICE" \
    --reconstruction-only --output "$OUTPUT" "${extra[@]}"
}
run_entry
