#!/usr/bin/env bash
# A/B/C 独立评测：--variant 选 A、B 或 C；在仓库根目录执行。
# bash scripts/launch/submit_replogle_control_eval.sh --variant A --dry-run
# bash scripts/launch/submit_replogle_control_eval.sh --variant B --dry-run
# bash scripts/launch/submit_replogle_control_eval.sh --variant C --dry-run
# --dry-run 换成 --submit 即提交；加 --node none 不固定节点；全部参数见 --help。

set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
baseline_art="$project_dir/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
export PYTHONPATH="$baseline_art/environment/python_packages:$baseline_art/source_overlay:$project_dir/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${SC_EVAE_PYTHON:-python3}" \
    scripts/launch/submit_replogle_control_eval.py "$@"
