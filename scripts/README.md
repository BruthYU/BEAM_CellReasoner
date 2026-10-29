# VQ-VAE实验入口

一类任务一个Bash入口，参数通过环境变量覆盖；Python实现位于 `lib_vqvae/` 和 `src/celltok/`。
不提交按日期命名的启动器、单独的smoke配置或旧实验输出。

| 入口 | 职责 | Python实现 |
| --- | --- | --- |
| `train.sh` | 准备数据，调用model.fit_ddp_advanced | `celltok.train` |
| `data.sh` | H5AD转换、完整分片清单冻结、归档校验 | `celltok.data` |
| `evaluate.sh` | 固定测试协议的流式重建评测 | `celltok.evaluation` |
| `test.sh` | CPU回归，可选多进程等价性检查 | `tests/`、`celltok.checks` |

`_common.sh`统一处理环境、tmux和rlaunch；默认 `--launch`，测试默认 `--local`。
`--dry-run`只打印命令。每个新实验应指定新的 `OUTPUT_DIR` 和 `SESSION`。

## 训练设置

| 参数 | 默认 |
| --- | --- |
| GPUS / CPUS / MEMORY_MIB | 4 / 32 / 128000 |
| CONFIG | configs/tokenizer_stage1_full_primary.yaml |
| MANIFEST | data/stage1/full_primary/fit_full_primary_20261026.csv |
| OPTIMIZER / SCHEDULER | adamw / warmup_cosine |
| LEARNING_RATE / EPOCHS | 0.0003 / 50 |
| BATCH_SIZE / WORKERS | 全局512 / 每rank 2 |
| SAVE_EVERY | 每5轮和最后一轮 |
| CACHE_DIR | /tmp/celltok-training |

AdamW采用FP32、betas=(0.9,0.999)、eps=1e-8；weight decay=5e-4，warmup比例0.03，
最低LR比例0.1。4卡时每卡128，单卡时512；总batch必须能被rank数整除。
DDP使用rank分片的archive-aware sampler，同步梯度后每rank执行一步，不额外除以卡数。
为保证完整batch且各rank步数相等，每个归档尾部及每轮未凑齐rank的批次会被丢弃，
所以每轮实际消费细胞数可能小于manifest总数；`run_manifest.json`记录实际steps。

训练定期评估每rank的第一个validation batch并归约，保持原fit的轻量监测方式；
这不是全量验证集指标。模型的正式比较使用独立的 `evaluate.sh`。

```bash
# 单卡：保持全局batch 512
GPUS=1 OUTPUT_DIR=outputs/tokenizer/single SESSION=celltok-single bash scripts/train.sh

# SGD对照：相同数据和loader，切换优化器/调度器
OPTIMIZER=sgd LEARNING_RATE=0.0001 SCHEDULER=reduce_on_plateau EPOCHS=30 \
  OUTPUT_DIR=outputs/tokenizer/sgd SESSION=celltok-sgd bash scripts/train.sh

# 同一入口的限步检查；不需要另写smoke脚本/配置
EPOCHS=2 MAX_STEPS=2 SAVE_EVERY=1 OUTPUT_DIR=outputs/tokenizer/check \
  SESSION=celltok-train-check bash scripts/train.sh

# 完整epoch checkpoint恢复；保持原总epoch、配置、卡数和loader设置
RESUME=outputs/tokenizer/my_run/epoch_20/model.pth OUTPUT_DIR=outputs/tokenizer/my_run \
  SESSION=celltok-resume bash scripts/train.sh
```

`VQVAE.fit`是原单卡SGD兼容接口；`fit_ddp_advanced`是本次正式训练入口。
SGD选项用于对照，不代表与作者公开权重具有完全相同的训练设置。

## CPU与DDP检查

```bash
bash scripts/test.sh
DDP_RANKS=2 bash scripts/test.sh
```

第二条额外运行2进程CPU gloo检查：将DDP梯度、优化器更新与独立全局batch计算比较，
并检查参数同步、checkpoint和进度输出，不申请GPU。

已有GPU worker内可检查NCCL：

```bash
PYTHONPATH=src:. torchrun --standalone --nproc_per_node=4 -m celltok.checks \
  --expected-world-size 4 --device cuda --output outputs/checks/ddp_gpu
```

## 集群与监控

本项目所有rlaunch（包括GPU=0）只能使用 `CHARGED_GROUP=beam_gpu_pool`。
镜像和挂载可通过 `IMAGE`、`MOUNT_SOURCE`、`MOUNT_TARGET` 覆盖；仓库必须位于挂载目录下。
默认镜像为 `celltok-py312-torch251-cu121-cellstream010-v2`。
其他集群用户需要填写自己的挂载路径；数据和模型路径通过配置或环境变量传入。

```bash
tmux attach -t celltok-my-run
tail -f logs/celltok-my-run.worker.log
cat outputs/tokenizer/my_run/progress.json
```

tmux运行于集群，电脑断网不影响worker；任务结束后rlaunch退出释放资源。
有意中止时须确认worker也已退出。训练前会拒绝覆盖已有run，恢复需显式提供 `RESUME`。
