# Replogle 代码副本的安装与准备

本目录用于代码审阅。正式实验仍在 `/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue/sc-evae` 运行。
`datasets/`、`artifacts/`、`outputs/`、实验 `docs/` 不进入 Git；仅复制代码不能直接续训或复查历史模型。

## 仅检查代码、预览任务（CPU，无需模型）

在本目录执行：

```bash
python3 scripts/checks/check_script_layout.py
bash scripts/launch/submit_replogle_qwen.sh --preset original-control1 --eval-after-training --dry-run
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-c --control-cells-nums 8 --dry-run
```

Qwen **训练提交器**的 `--help`、`--dry-run` 只需 Python 标准库；不读模型、不创建请求、不提交任务。
`--check`、`--spec`、`--submit` 会验证外部资产、模型上下文长度及冻结输入哈希，不能用预览通过代替训练准备通过。
独立评测提交器需要真实 `--run-dir` 才能选择 checkpoint，因此其 dry-run 仍需该运行的元数据。
三个公共 Qwen/对照评测 Bash 提交入口默认使用当前环境的 `python3`，可用 `SC_EVAE_PYTHON` 指定解释器。

## 安装环境

原 AR/Stage 1 的环境定义为 `environment.yml`，锁文件为 `conda-linux-64.lock.txt` 和 `requirements-linux-64.lock.txt`。
Qwen 另需 `pyproject.toml` 的 `qwen` 可选依赖；基础 AR 安装不会自动安装 Transformers/FLA。
以下适用于 Linux x86_64、Python 3.12、CUDA 12.8：

```bash
conda env create -f environment.yml
conda activate sc-evae
python -m pip install '.[qwen]'
python -m pip check
python scripts/environment/check_environment.py
python -c 'from transformers import Qwen3_5ForCausalLM; import fla; import triton'
```

Qwen 版本来自既有实验记录；Triton 使用 Torch 2.9.1 声明要求的 **3.5.1**。
旧实验的 `environment/python_packages` 覆盖安装了 Triton 3.7.1，不能把这套覆盖目录原样当作可由 pip 正常解析的锁文件。
在 H100/H200 上，现有 `install_fast_delta()` 明确要求 Triton ≥3.7.1，因此 **仅安装 `.[qwen]` 不足以运行 Hopper 训练**。继续采用既有实验的独立覆盖目录，不修改基础 Torch 环境：

```bash
qwen_overlay="$PWD/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1/environment/python_packages"
python -m pip install --no-deps --target "$qwen_overlay" -r scripts/environment/requirements-qwen-hopper-overlay.txt
export PYTHONPATH="$qwen_overlay:$PWD/scripts"
python -c 'import _bootstrap; from replogle_llm_common import install_fast_delta; install_fast_delta()'
```

此处 `--no-deps` 是有意隔离的 kernel 覆盖，与已有实验一致；不是一套满足所有包声明的单环境锁定。基础环境的 `pip check` 不证明覆盖后的运行组合依赖一致。
只在新建或已核验的覆盖目录执行，避免混入其他版本；不要盲目复制旧覆盖目录的全部包。
新安装路径尚需实际 GPU smoke 验证；CPU 导入检查不验证 CUDA kernel 或训练速度。

worker 默认在镜像的 `/opt/sc-evae` 环境运行。原镜像只保证基础 AR 依赖，Qwen 需要部署上述依赖或经核验的既有覆盖环境；不要仅在提交机安装后就认为 worker 已具备依赖。
当前提交器使用固定的既有集群镜像摘要；自建新镜像需显式更新所用提交器的镜像配置。

## 外部资产（训练、评测前必须准备）

以下路径相对本目录；现有预设使用这些冻结实验路径，并非通用下载器。建议继续在完整原运行目录执行实验。
若要迁移运行环境，必须完整迁移所选预设的依赖并核验路径/哈希，不能只复制一个模型 `config.json`。

| 资产 | 位置 / 用途 |
| --- | --- |
| Replogle 原始 H5AD | `datasets/replogle/`；既有原始数据位于 `/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig` |
| 条件文本与细胞索引 | `artifacts/replogle_text_conditions/v1/`，包括 `conditions.jsonl`、索引和 manifest |
| 冻结 NB tokenizer | `outputs/replogle_experiments/ar-fsq-nb-l64-k512-990fc9c87a-2gpu-0868/stage1/`，包括配置、统计量、checkpoint；评测还需要同实验 `stage1-smoke/deg_table.pt` |
| Qwen 原始权重 | `/mnt/shared-storage-gpfs2/beam-gpfs02/wangyue/BEAM_CellReasoner/model/LLM/Qwen3.5-0.8B`；导出器要求 `LOCAL_SOURCE.json` 和其中列出的权重、tokenizer 文件 |
| 语言模型导出与 FSQ 缓存 | `artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1/`：`initial_text_model/`、`cache/`、划分数组、`sources.json`、检查及来源清单 |
| A/B 数据与检查清单 | `artifacts/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-v1/`，含划分、技术分组、对照池信息、验证索引与 `ready_inputs.json` |
| 原划分单对照 | `artifacts/replogle_llm_control/qwen35-08b-fsq-nb-original-split-control1-v1/`；还依赖基础缓存及 A/B 技术分组元数据 |
| C 实验 | `artifacts/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-c-v1/`；数据使用 A/B 划分，另有 C 的检查清单 |
| 独立评测 | 对应训练运行的 request/run_config、验证最优 checkpoint 和所选评测协议的 plan/清单；不随代码分发 |

准备流程按依赖顺序为：文本表 → 原划分/来源清单 → NB token 缓存与 Qwen 语言权重导出 → CPU 输入检查 → 各对照预设准备和检查。
入口分别在 `scripts/data/prepare_replogle_text_conditions.py`、`prepare_replogle_llm.py`、`cache_replogle_nb_tokens.py`、`prepare_replogle_llm_model.py` 以及 `scripts/checks/`。
A/B 使用 `prepare_replogle_control_ab.py` 和 `check_replogle_control_ab.py`；单对照使用 `prepare_replogle_control_baseline.py`；C 使用 `check_replogle_control_c.py`。

这些准备程序保留了历史实验约束：指定的 NB run、原始模型路径/来源收据、冻结的细胞数，以及缓存程序的 GPU 节点检查；单对照准备还读取历史原文本训练的 `formal/run_config.json`。
**它们不是可在任意空目录顺序运行的一键初始化流程。** 迁移时先核对配置和来源文件中的绝对路径；不要覆写原实验的冻结清单来消除报错。
旧 `source_overlay/sc_evae` 可能链接到原目录；新环境应安装本代码包，不能盲目复制该链接。

准备完成后用 `--check` 验证，只有显式 `--submit` 才提交 rjob。新结果写入独立 `outputs/replogle_qwen/<preset>/<run-id>/`。

## 镜像构建

Dockerfile 使用 conda-pack 归档作为构建输入，该大文件不进入 Git。先在完整且没有 editable install 的 conda 环境生成它：

```bash
bash scripts/environment/pack_environment.sh
docker build -t sc-evae:local .
```

默认输入为 `artifacts/sc-evae-py312-torch291-cu128-linux-x86_64.tar.gz`，`.dockerignore` 明确允许该文件进入构建上下文。
`pack_environment.sh`、`unpack_environment.py`、`check_environment.py` 均随代码提交。
Qwen 镜像需要先在被打包环境安装 `.[qwen]`；Hopper 的 Triton 覆盖目录保留在项目共享存储，由 worker 的 `PYTHONPATH` 加载，不混入 conda-pack 基础环境。基础环境的检查通过不等于 Qwen GPU 验证通过。
构建和运行还需要集群基础镜像访问权限、合适的 NVIDIA 驱动及 GPU 调度权限。

## 代码校验与历史审计

默认 `check_script_layout.py` 检查当前脚本哈希、迁移后路径、Python/Shell 语法、导入与文档链接，不要求带入历史归档。
修改已登记脚本时，需要在审阅代码差异后更新 `scripts/layout.json` 中的 `new_sha256` / `shared_files` 当前哈希；保留 `original_sha256` 不变。

```bash
# 仅在具备历史归档和原实验元数据的完整运行目录执行
python3 scripts/checks/check_script_layout.py --historical
```

历史审计额外需要 `layout.json` 中每个 `archive` 路径、`artifacts/repository_cleanup/20261123_scripts_layout/experiment_metadata_before.json` 及其引用的实验元数据。
缺少时明确失败，不视为验证通过。训练沿用历史 manifest 时仍使用同样严格的原始归档/当前文件校验，不因默认代码检查独立而放宽。
