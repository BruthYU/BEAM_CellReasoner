#!/usr/bin/env bash
# Train only a prior: --head nb --vae-path /path/to/trained/stage1 --submit.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$project_dir/scripts/launch/run_replogle_experiment.sh" "$@" --stage stage2
