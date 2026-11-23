#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run_id="${3:?Run ID required}"
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
exec bash "$project_dir/scripts/workers/run_replogle_qwen_worker.sh" "${1:?}" "${2:?}" \
    "$project_dir/outputs/replogle_llm_control/qwen35-08b-fsq-nb-original-split-control1-v1/$run_id"
