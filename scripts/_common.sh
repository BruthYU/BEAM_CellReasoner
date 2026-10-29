#!/usr/bin/env bash
# Shared environment and tmux/rlaunch handling; experiment parameters live in callers.
set -euo pipefail
ENTRY_PATH="$(realpath "${BASH_SOURCE[1]}")"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_MODE=disabled
export TZ=Asia/Hong_Kong
IMAGE="${IMAGE:-registry.h.pjlab.org.cn/ailab-beam-beam_gpu/aivc-celltok:celltok-py312-torch251-cu121-cellstream010-v2}"
MOUNT_SOURCE="${MOUNT_SOURCE:-gpfs://gpfs2/beam-gpfs02/wangyue}"
MOUNT_TARGET="${MOUNT_TARGET:-/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue}"
MODE="${1:---launch}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN=python
  if [[ "$MODE" == --local && "${GPUS:-0}" == 0 && -x "$REPO_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python"
  fi
fi

run_python() {
  if [[ "$MODE" == --dry-run ]]; then
    printf '%q ' "$PYTHON_BIN" "$@"; printf '\n'
  else
    "$PYTHON_BIN" "$@"
  fi
}

run_entry() {
  case "$MODE" in
    --local|--dry-run) experiment ;;
    --launch)
      if [[ "$CHARGED_GROUP" != beam_gpu_pool ]]; then
        echo "User resource policy: rlaunch must use CHARGED_GROUP=beam_gpu_pool (including CPU workers); no quota-group fallback." >&2
        return 2
      fi
      if declare -F preflight >/dev/null; then preflight; fi
      case "$REPO_DIR/" in
        "$MOUNT_TARGET/"*) ;;
        *) echo "Repository must be under MOUNT_TARGET=$MOUNT_TARGET for the worker to access this checkout." >&2; return 2 ;;
      esac
      command -v tmux >/dev/null
      command -v rlaunch >/dev/null
      if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session already exists: $SESSION. Use tmux attach -t $SESSION" >&2
        return 1
      fi
      mkdir -p logs
      local worker_command launch_string
      printf -v worker_command 'cd %q && bash %q --local' "$REPO_DIR" "$ENTRY_PATH"
      local -a launch=(rlaunch --gpu="$GPUS" --cpu="$CPUS" --memory="$MEMORY_MIB"
        --charged-group="$CHARGED_GROUP" --max-wait-duration=600m
        --image="$IMAGE"
        --mount="$MOUNT_SOURCE:$MOUNT_TARGET")
      if [[ -n "${SOURCE_MOUNT:-}" ]]; then launch+=("--mount=$SOURCE_MOUNT"); fi
      local name
      for name in GPUS CPUS MEMORY_MIB CHARGED_GROUP SESSION "${PARAM_NAMES[@]}"; do
        launch+=("--env=$name=${!name}")
      done
      launch+=(-- bash -c "$worker_command")
      printf -v launch_string '%q ' "${launch[@]}"
      tmux new-session -d -s "$SESSION" "set -o pipefail; $launch_string 2>&1 | tee '$REPO_DIR/logs/$SESSION.worker.log'"
      echo "Submitted: tmux attach -t $SESSION"
      ;;
    *) echo "Usage: bash $ENTRY_PATH [--launch|--local|--dry-run]" >&2; return 2 ;;
  esac
  # These are executable entrypoints. Exit inside the parsed function so Bash
  # does not resume reading a live-edited caller after a long experiment ends.
  exit 0
}
