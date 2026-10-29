#!/usr/bin/env bash
# One entry for Stage1 conversion, validation and freezing a complete manifest.
GPUS="${GPUS:-0}"; CPUS="${CPUS:-8}"; MEMORY_MIB="${MEMORY_MIB:-64000}"
CHARGED_GROUP="${CHARGED_GROUP:-beam_gpu_pool}"; SESSION="${SESSION:-celltok-data}"
TASK="${TASK:-validate}"
MANIFEST="${MANIFEST:-data/stage1/full_primary/fit_full_primary_20261026.csv}"
GENES="${GENES:-data/5000_gene_names.csv}"
INPUT="${INPUT:-}"; OUTPUT="${OUTPUT:-outputs/data/validation.json}"
CELL_SELECTION="${CELL_SELECTION:-primary}"; BLOCK_ROWS="${BLOCK_ROWS:-8192}"
SHARD_MANIFEST="${SHARD_MANIFEST:-}"; CACHE_DIR="${CACHE_DIR:-/tmp/celltok-validation}"
# TASK=convert INPUT=/path/study.h5ad OUTPUT=/path/study.csad
# TASK=freeze MANIFEST=/path/source.csv SHARD_MANIFEST=/path/shards.csv OUTPUT=/path/complete.csv
PARAM_NAMES=(TASK MANIFEST GENES INPUT OUTPUT CELL_SELECTION BLOCK_ROWS SHARD_MANIFEST CACHE_DIR)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
experiment() {
  case "$TASK" in
    convert)
      [[ -n "$INPUT" ]] || { echo "INPUT is required for convert" >&2; return 2; }
      run_python -m celltok data convert --input "$INPUT" --output "$OUTPUT" \
        --genes "$GENES" --matrix raw-or-X --storage counts --cell-selection "$CELL_SELECTION" --block-rows "$BLOCK_ROWS"
      ;;
    validate)
      run_python -m celltok data validate --manifest "$MANIFEST" --genes "$GENES" \
        --local-cache-dir "$CACHE_DIR" --report "${OUTPUT%.json}.csv" --summary "$OUTPUT"
      ;;
    freeze)
      [[ -n "$SHARD_MANIFEST" ]] || { echo "SHARD_MANIFEST is required for freeze" >&2; return 2; }
      run_python -m celltok data freeze --manifest "$MANIFEST" --shard-manifest "$SHARD_MANIFEST" \
        --output "$OUTPUT" --require-all
      ;;
    *) echo "TASK must be convert, validate or freeze" >&2; return 2 ;;
  esac
}
run_entry
