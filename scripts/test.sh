#!/usr/bin/env bash
# CPU tests with synthetic tensors/archives; no GPU or full-data run is allocated.
GPUS="${GPUS:-0}"; CPUS="${CPUS:-4}"; MEMORY_MIB="${MEMORY_MIB:-8192}"
CHARGED_GROUP="${CHARGED_GROUP:-beam_gpu_pool}"; SESSION="${SESSION:-celltok-tests}"
DDP_RANKS="${DDP_RANKS:-0}"; OUTPUT_DIR="${OUTPUT_DIR:-outputs/checks}"
# Optional small-tensor gloo check, not a new multi-GPU training experiment:
# DDP_RANKS=2
PARAM_NAMES=(DDP_RANKS OUTPUT_DIR)
if [[ $# == 0 ]]; then set -- --local; fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
experiment() {
  run_python -m pytest tests -q
  if [[ "$DDP_RANKS" != 0 ]]; then
    run_python -m torch.distributed.run --standalone --nproc_per_node="$DDP_RANKS" \
      -m celltok.checks --device cpu --expected-world-size "$DDP_RANKS" --output "$OUTPUT_DIR/ddp_cpu"
  fi
}
run_entry
