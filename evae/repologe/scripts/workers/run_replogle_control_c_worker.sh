#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_replogle_control_ab_worker.sh" C "${2:?}" "${3:?}" "${4:?}" "${1:?}" "${5:-2}"
