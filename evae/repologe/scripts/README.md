# 脚本导航

先阅读 [安装与外部资产](../SETUP.md)。`--dry-run` 只预览配置，`--check`/`--submit` 需要完整实验资产与 Qwen 环境。

从 `launch/` 选择任务入口，通常不需要直接调用 worker 或内部模块。以下命令均在仓库根目录执行。

| 目录 | 用途 |
| --- | --- |
| launch/ | 提交 rjob、预览资源配置；日常主要入口 |
| workers/ | rjob 内部执行流程，串联检查、训练和评测 |
| training/ | Qwen/对照输入训练程序 |
| evaluation/ | Qwen/对照输入的生成、合并及九指标评测 |
| data/ | 数据划分、文本模板、FSQ 缓存、预训练模型准备 |
| checks/ | 正式流程仍依赖的数据、checkpoint、生成和指标检查 |
| analysis/ | loss 曲线、结果汇总、单细胞 case 展示 |
| lib/ | 数据集、配置、生成与实验管理的共享实现 |
| environment/ | conda、worker 环境及镜像构建辅助工具 |

根目录的 `train.py`、`eval.py`、`eval_pfizer.py` 是作者原始入口，保留原路径和内容。`_bootstrap.py` 统一直接执行脚本时的导入路径；`layout.json` 记录此次旧路径到新路径的迁移与文件哈希。

## Qwen 统一入口

新实验统一使用 [launch/submit_replogle_qwen.sh](launch/submit_replogle_qwen.sh)，用 `--preset` 选择数据划分和输入模板。五种预设都复用同一份 [训练循环](training/train_replogle_qwen.py)、[worker](workers/run_replogle_qwen_worker.sh) 和 [分卡评测程序](evaluation/eval_replogle_qwen.py)。默认继续使用冻结的 NB tokenizer，从原始 Qwen 权重开始全量 SFT。

| --preset | 数据划分 | 输入 | 对照数量 |
| --- | --- | --- | --- |
| original-text | 原 Qwen 划分 | 原文本模板 | 0 |
| original-control1 | 原 Qwen 划分 | 原文本＋Control Cell | 1 |
| hepg2-a | A/B/C 的组合留出＋整 HepG2 留出 | A 文本模板 | 0 |
| hepg2-b | 同 A | B 文本＋Control Cell | 1 |
| hepg2-c | 同 A | C 文本＋Control Cells | `--control-cells-nums`，默认 8 |

模板保留原区别：原划分的两个预设没有额外 `Predicted Cell:` 文本，A/B/C 有；C 另外带对照数量。C 的 N=1 仍是 C 模板，不会悄悄切换成 B。对照采样和输出标签继续使用原有 dataset 类。

```bash
# 校验并预览：原划分＋1 control，训练结束后自动评测九项指标
bash scripts/launch/submit_replogle_qwen.sh \
  --preset original-control1 --eval-after-training --check

# C：参数化对照数量、GPU、每卡 batch、全局 batch；默认只打印命令
bash scripts/launch/submit_replogle_qwen.sh \
  --preset hepg2-c --control-cells-nums 8 \
  --gpus 4 --per-device-batch-size 8 --global-batch-size 128 \
  --cpus 32 --memory-gib 128 --charged-group beam_gpu \
  --eval-after-training --dry-run

# 要固定节点时加 --node HOST；要实际提交时加 --submit
bash scripts/launch/submit_replogle_qwen.sh --help
```

公共入口的资源默认值为 **8 卡、每卡 batch 16、全局 batch 128、48 CPU、195 GiB、beam_gpu、不固定节点**。梯度累计自动计算为 `global_batch / (gpus × per_device_batch)`，不能整除会报错。学习率、epoch 和保存频率沿用对应配置文件，也可用 `--learning-rate`、`--epochs`、`--save-steps` 覆盖。C 的长输入是否适合更大的 batch 仍需 GPU benchmark 确认。

新提交的输出统一进入 `outputs/replogle_qwen/<preset>/<run-id>/`，请求记录进入 `artifacts/replogle_qwen/<preset>/`。相同输出目录不能重复使用。`--check` 和默认 dry-run 不创建任务、不预留请求；只有 `--submit` 会提交 rjob。每个新请求记录当前脚本哈希，worker 会在执行前复核。

`--eval-after-training` 在同一个 rjob 内串联 smoke → 正式训练 → 验证 CE 最优 checkpoint → 九指标评测；不加时仅训练。评测默认使用全部申请的 GPU，可用 `--eval-gpus` 降低，用 `--eval-batch-size` 控制每卡生成 batch（默认 64）。各 GPU 生成互不重叠的细胞，合并完整群体后才算指标。原划分评测内部验证和论文测试，A/B/C 评测组合验证、组合测试、论文 HepG2 和全量 HepG2，并保留逐细胞系结果与宏平均。NB 深度继续采用实测目标细胞的 2000-HVG 总计数。

统一评测使用现有单对照流程的 `seed + global_batch_id` 随机流和 float32 pseudobulk PDS 校正。因此，新评测不会承诺与历史单 GPU A/B/C 的随机生成逐位相同；历史预测、指标和表格没有重算或改写。

## 原划分 Qwen 单独提交评测

```bash
bash scripts/launch/submit_replogle_qwen_eval.sh --preset original-control1 --run-dir RUN_DIR --dry-run
bash scripts/launch/submit_replogle_qwen_eval.sh --preset original-text --run-dir RUN_DIR --gpus 2 --dry-run
```

`RUN_DIR` 是已有训练运行目录，也可传其 `formal/`；旧文本模型可直接传 `outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1/formal`。`--dry-run` 换成 `--submit` 才提交；`--check` 只验证数据和最佳 checkpoint。默认 1 GPU、16 CPU、96 GiB、beam_gpu、不固定节点；支持 `--gpus 1|2|4|8`、`--batch-size`、`--cpus`、`--memory-gib`、`--charged-group`、`--node`。

只读取验证最优模型，评测原划分内部验证和论文测试的九项指标，不启动训练。结果单独写入 `outputs/replogle_qwen_evaluation/<preset>/<eval-id>/evaluation/`，不覆盖训练目录原有结果。评测仍复用 `run_replogle_qwen_eval.sh` 和公共引擎。两种模板统一采用已有多卡评测的全局 batch 随机流与精确 float32 PDS 口径；旧文本模型历史单卡评测使用的生成顺序不同，因此新分片评测不保证复现历史表格的每个随机数值。

## A/B/C 独立评测

A、B、C 共用 [launch/submit_replogle_control_eval.sh](launch/submit_replogle_control_eval.sh)，每次用 `--variant` 明确选择一个实验。A/B 不会捆绑运行；三者的模型和结果目录相互独立。

```bash
# 三条命令分别预览 A、B、C 的独立任务，都不提交
bash scripts/launch/submit_replogle_control_eval.sh --variant A --dry-run
bash scripts/launch/submit_replogle_control_eval.sh --variant B --dry-run
bash scripts/launch/submit_replogle_control_eval.sh --variant C --dry-run
```

这些命令复查已冻结的历史 checkpoint，沿用各自的评测集合和协议。生成 batch 用 `--batch-size`（默认 64）；节点用 `--node HOST` 或 `--node none`。保留历史资源默认值：1 GPU、16 CPU、96 GiB、beam_gpu、0863 节点；实际运行需加 `--submit`。

worker 统一调用 `evaluation/eval_replogle_qwen.py --protocol legacy-control --variant A|B|C`。历史 A/B 使用 CSV 审计 PDS，C 使用 float32 pseudobulk 审计，均按原协议保留。新版训练后多卡评测使用默认 `--protocol sharded`；原文本版 quick/validation/test 复查使用 `--protocol legacy-text`。模型加载、FSQ 采样和 observed-depth NB 解码共用 `lib/replogle_qwen_inference.py`，九指标共用 `lib/replogle_qwen_metrics.py`。旧 evaluation 兼容文件现已归档，`evaluation/` 仅保留 `eval_replogle_qwen.py`；检查程序和历史文本训练编排已直接使用公共模块与入口。

## 兼容入口及 AR

| 入口 | 当前用途 |
| --- | --- |
| [submit_replogle_control_eval.py](launch/submit_replogle_control_eval.py) | 历史固定 checkpoint 的 A/B/C 评测复查；保留旧协议和旧节点默认值 |
| [submit_replogle_llm.sh](launch/submit_replogle_llm.sh) | 历史文本版学习率搜索编排；训练内核已共用。新正式实验用 original-text |
| [run_replogle_stage1.sh](launch/run_replogle_stage1.sh) | NB/MSE tokenizer Stage 1 |
| [run_replogle_stage2.sh](launch/run_replogle_stage2.sh) | 原 AR Stage 2 |
| [run_replogle_experiment.sh](launch/run_replogle_experiment.sh) | 参数化 AR 实验 |

旧训练脚本保留参数解析适配，训练循环已删除并转到公共实现。历史 A/B/C worker 和固定模型评测入口用于读取旧任务记录；不要将其视为新实验默认入口。AR 与 LLM 模型不同，继续独立维护。

```bash
# 代码副本的目录、语法、链接与当前哈希；不依赖历史归档
python3 scripts/checks/check_script_layout.py
# 在完整运行目录中才使用 --historical 额外校验历史归档和实验元数据

# 在项目 conda＋既有 python_packages/source_overlay 环境运行 CPU 回归
python scripts/checks/check_replogle_qwen.py
```

本次合并前快照位于 `artifacts/repository_cleanup/20261123_qwen_consolidation/`。CPU 检查覆盖五种预设的 prompt/label 一致性、对照采样、评测行集合、1/2/4/8 卡分片合并和九指标合成样例。未为脚本合并提交 GPU 任务；GPU 端完整训练和生成尚未重新验证。

## 2026-11-23 清理范围

从工作目录删除了 11 个一次性文件：

- 已取消的文本 embedding 路线：编码程序、提交脚本、worker，共 3 个。文本模板准备仍保留，LLM 训练继续使用。
- 已完成的 C 换卡续训与交接：固定任务 coordinator、续训工具、准备/检查/训练/提交/worker，共 7 个。
- 未被现有入口引用的固定双卡 DDP 检查，共 1 个。

其余 `check_*`、`validate_*` 保留在 `checks/`，因为训练前检查、GPU smoke、冻结输入校验和九指标验证仍调用它们。GPU smoke 是正式任务中的短检查阶段，不另外复制一套实验脚本。

迁移前的脚本快照保存在 `artifacts/repository_cleanup/20261123_scripts_layout/original/scripts/`（仅运行目录提供），便于追溯旧任务。它们不再是可直接提交的入口；不要复制历史固定任务的交接脚本重新运行。工作目录没有保留一排旧路径软链接，日常使用上述新路径。

旧 `artifacts/` 和 `outputs/` 中的清单、配置、checkpoint、loss、评测缓存与结果未改写。旧清单中的脚本路径通过显式迁移记录校验：先验证快照确实符合旧哈希，再验证新脚本符合迁移后的哈希；没有关闭哈希检查，也没有把新代码伪装为旧哈希。

不要按目录名批量删除 `stage1-smoke`：当前正式 NB 评测仍使用其中的 `deg_table.pt`。同理，`cpu_preflight.json` 是校验记录，不能因为名字带 check/smoke 就当作无用文件删除。

结果与实验定义保留在原运行目录的 `docs/README.md`，不随代码副本提交。

## 评测兼容入口归档

2026-11-23 将 7 个已被统一入口替代的兼容文件移出活动目录：4 个旧评测 Python 入口、C 专用提交/worker、单对照旧评测 Bash。归档在 `artifacts/repository_cleanup/20261123_eval_entry_cleanup/scripts/`，仅供查阅。所有活动引用、准备清单和文档链接已迁移。

- A/B/C 独立提交：`bash scripts/launch/submit_replogle_control_eval.sh --variant A|B|C`。
- 训练后评测或恢复分片：`bash scripts/workers/run_replogle_qwen_eval.sh RUN_DIR`。
- 原文本 quick/validation/test：`python scripts/evaluation/eval_replogle_qwen.py --protocol legacy-text --model-dir MODEL --output-dir OUTPUT --split quick|validation|test`。

本次只归档兼容代码，没有删除模型、FSQ 缓存、数据划分或评测结果。训练端适配入口仍有调用关系，本次保留。

原划分单对照的 3 个旧提交入口已归档至 `artifacts/repository_cleanup/20261123_baseline_launch_cleanup/`。统一使用 `submit_replogle_qwen.sh --preset original-control1`；训练后评测再加 `--eval-after-training`。

A/B、C 的两个旧训练提交 Bash 已归档至 `artifacts/repository_cleanup/20261123_ab_c_launch_cleanup/`。统一使用 `submit_replogle_qwen.sh --preset hepg2-a|hepg2-b|hepg2-c`，C 的对照数量继续用 `--control-cells-nums N`。
