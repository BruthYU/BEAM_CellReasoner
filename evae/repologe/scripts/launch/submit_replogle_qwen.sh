#!/usr/bin/env bash
# 在仓库根目录调用；--dry-run 预览，换成 --submit 提交。
# 预设：original-text（原文本）、original-control1（原划分+1对照）、hepg2-a/b/c（A/B/C）。

# # 只训练
# bash scripts/launch/submit_replogle_qwen.sh --preset original-control1 --dry-run

# # 训练后评测
# bash scripts/launch/submit_replogle_qwen.sh --preset original-control1 --eval-after-training --dry-run

# # A
# bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-a --dry-run
# bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-a --submit

# # B
# bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-b --dry-run
# bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-b --submit


# # C
# bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-c --control-cells-nums 8 --dry-run
# bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-c --control-cells-nums 8 --submit



set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
baseline_art="$project_dir/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
export PYTHONPATH="$baseline_art/environment/python_packages:$baseline_art/source_overlay:$project_dir/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${SC_EVAE_PYTHON:-python3}" \
    scripts/launch/submit_replogle_qwen.py "$@"
