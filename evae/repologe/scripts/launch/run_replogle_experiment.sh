#!/usr/bin/env bash
# Example: bash scripts/launch/run_replogle_experiment.sh --prior ar --quantizer fsq --head nb --submit
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1
if [[ -n "${SC_EVAE_PYTHON:-}" ]]; then
    python_bin="$SC_EVAE_PYTHON"
elif [[ -x /opt/sc-evae/bin/python ]]; then
    python_bin=/opt/sc-evae/bin/python
elif [[ -x /mnt/shared-storage-user/wangyue/conda/envs/sc-evae/bin/python ]]; then
    python_bin=/mnt/shared-storage-user/wangyue/conda/envs/sc-evae/bin/python
else
    python_bin=python
fi
exec "$python_bin" scripts/lib/replogle_experiment.py "$@"
