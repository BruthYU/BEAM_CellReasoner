#!/usr/bin/env bash
# Run with the sc-evae conda environment active. No editable installs allowed.
set -euo pipefail
export PYTHONNOUSERSITE=1

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
: "${CONDA_PREFIX:?Activate the sc-evae conda environment first}"
python_bin="$CONDA_PREFIX/bin/python"
output="${1:-$project_dir/artifacts/sc-evae-py312-torch291-cu128-linux-x86_64.tar.gz}"
mkdir -p -- "$(dirname -- "$output")"

"$python_bin" -m pip check
"$python_bin" scripts/environment/check_environment.py
"$python_bin" -m conda_pack.cli -p "$CONDA_PREFIX" -o "$output" \
    --compress-level 1 --n-threads 4
(cd -- "$(dirname -- "$output")" && sha256sum "$(basename -- "$output")") \
    > "$output.sha256"
printf 'Environment archive: %s\n' "$output"
