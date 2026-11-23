#!/usr/bin/env bash
# 单独申请评测任务；RUN_DIR 为已有训练目录（也可传 formal 目录）。
# bash scripts/launch/submit_replogle_qwen_eval.sh --preset original-control1 --run-dir RUN_DIR --dry-run
# bash scripts/launch/submit_replogle_qwen_eval.sh --preset original-text --run-dir RUN_DIR --gpus 2 --dry-run
# --dry-run 换成 --submit 提交；--check 只校验。默认1卡、16CPU、96GiB、beam_gpu、不固定节点。
# --batch-size 控制每卡生成batch；--node HOST 可固定节点。全部参数见 --help。

set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
baseline_art="$project_dir/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1"
export PYTHONPATH="$baseline_art/environment/python_packages:$baseline_art/source_overlay:$project_dir/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${SC_EVAE_PYTHON:-python3}" \
    scripts/launch/submit_replogle_qwen_eval.py "$@"
