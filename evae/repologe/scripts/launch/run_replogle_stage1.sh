#!/usr/bin/env bash
# Train only a tokenizer, e.g. --head nb --submit. Hyperparameters have defaults.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$project_dir/scripts/launch/run_replogle_experiment.sh" "$@" --stage stage1
