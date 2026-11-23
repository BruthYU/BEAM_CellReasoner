# ExpressionVAE: Elucidating the Design Space of Generative Models for Single-Cell Perturbation Prediction

Sanjukta Bhattacharya, Christian Gensbigler, Shaamil Karim, Jon Lees

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.06.15.732063-b31b1b)](https://www.biorxiv.org/content/10.64898/2026.06.15.732063v1.abstract)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[Installation](#installation)** | **[Dataset Download](#dataset-download)** | **[Training](#training)** | **[Evaluation](#evaluation)** | **[Citation](#citation)**

---

`ExpressionVAE` is a scalar-quantized (FSQ) variational autoencoder that encodes
each cell's gene-expression profile as a fixed-length sequence of discrete codes, paired with a perturbation-conditioned generative prior over those codes (autoregressive, masked discrete diffusion, or flow matching). On both Replogle and Parse 1M datasets it reaches state-of-the-art on the distributional and mean based perturbation metrics. The frozen encoder transfers to an un-seen CRISPRi reversion benchmark introduced in the TeloHAEC dataset.

This repository contains everything needed to reproduce the paper's two datasets (Replogle, Parse 1M), the TeloHAEC evaluation, and the reported metrics.

## 本地实验导航

- **提交与脚本分类**：[scripts/README.md](scripts/README.md)。日常从 `scripts/launch/` 进入。
- **安装与外部资产**：[SETUP.md](SETUP.md)。代码副本不包含模型、缓存或历史实验文档；运行目录中的 `docs/README.md` 保留实验结果。
- **统一 Qwen 训练＋九指标流程**：`bash scripts/launch/submit_replogle_qwen.sh --preset original-control1 --eval-after-training --check`（只检查）；用 `--preset` 切换文本版、单对照及 A/B/C，用 `--submit` 实际提交。

2026-11-23 已按用途整理脚本，并移除已取消/已完成的一次性编码和换卡交接工具。历史实验的配置、权重与结果目录保持原位置。下面保留原项目说明及历次实验的操作背景；当前结果以 `docs/` 的结果表为准。

## Installation

Dependencies are managed with `[uv](https://docs.astral.sh/uv/)`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <this-repo-url> sc_evae
cd sc_evae
uv sync                 # runtime deps
```

### Conda environment and BEAM worker image (Linux x86_64)

The local Docker image `sc-evae:py312-torch291-cu128` has been built and validated
with Python 3.12.16, PyTorch 2.9.1+cu128, and CUDA 12.8. It is published at:

```text
registry.h.pjlab.org.cn/ailab-beam-beam_gpu/aivc-celltok:sc-evae-py312-torch291-cu128-v1
```

Registry manifest digest:
`sha256:5bcc353a9612286f6d2ee0567f06bfbff1b8094e258177472e375c1458af02e4`.
The registry image configuration matches the validated local image.

Launch it with the same gpfs2 mount and quota group as the CellReasoner workers:

```bash
tmux new-session -s sc-evae-worker
bash scripts/environment/launch_worker.sh
```

Inside the worker, the environment is already selected. To select it explicitly:

```bash
conda activate /opt/sc-evae
cd /mnt/shared-storage-gpfs2/beam-gpfs02/wangyue/sc-evae
python scripts/environment/check_environment.py --require-cuda
accelerate launch scripts/train.py \
    --config_path config/vaes/reploall-vae-fsq-mse.yaml
```

Keep datasets and training outputs under the mounted project directory. Exit the
worker after its work finishes. For a short GPU validation that exits
automatically, use `bash scripts/environment/launch_worker.sh --check`; `--dry-run` prints the
launch command. The default quota group is `beam_gpu_pool` and the default queue
wait is 10 minutes (`SC_EVAE_MAX_WAIT=600m` allows a longer wait).

The conda setup uses Python 3.12 and PyTorch 2.9.1 with CUDA 12.8.
Run these commands from the repository root:

```bash
conda env create -f environment.yml
conda activate sc-evae
python scripts/environment/check_environment.py
# On a GPU worker, also require a working CUDA device:
python scripts/environment/check_environment.py --require-cuda
```

For the exact versions used in the built images, recreate from the two lock
files instead of resolving `environment.yml` again:

```bash
conda create -n sc-evae --file conda-linux-64.lock.txt
conda activate sc-evae
export PYTHONNOUSERSITE=1
python -m pip install -r requirements-linux-64.lock.txt
python -m pip install --no-deps .
```

The project is installed as a regular wheel. After changing project source,
reinstall with `python -m pip install --no-deps .` in the environment.

To rebuild the worker image, the pack script prepares the Docker build input:

```bash
bash scripts/environment/pack_environment.sh
docker build -t sc-evae:py312-torch291-cu128 .
```

Publish rebuilds under a new registry tag using `docker tag` and `docker push`.
The Docker base is pinned to the PJLab Ubuntu 22.04 image. GPU execution requires
an allocated worker with a compatible NVIDIA driver. Build-node CPU validation
does not establish GPU training compatibility on a target worker.

The scLDM train/holdout split definitions (Replogle HepG2 = 372 perts; Parse
CD4-Naive = 27 cytokines) are JSON specs in `assets/` —
`assets/scldm_replogle_test.json` and `assets/scldm_parse1m_test.json`, each of
shape `{"cell_type": [...], "pert": [...]}`. A config selects one via
`dataset.split_path:`; point it at any JSON of the same shape to run a different
cell-type × perturbation holdout (no code change).

## Training

### Local Replogle Stage 1 run on GPFS2

The local recipe is `config/local/replogle-stage1-fsq-mse.yaml`, derived from
`config/vaes/reploall-vae-fsq-mse.yaml`. It uses the original loader, model,
loss and `scripts/train.py`. Four read-only H5AD files under
`/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/ECNU/remote_han/datasets/Replogle_Nadig`
are referenced through local symlinks with the basenames recognized by the loader.
No combined expression file is required.

The ordered 2,000-gene panel is pinned to scLDM commit
`0e03e404b096636a59493803d0bdc28ead9a3520`. The symbols and their unambiguous
Ensembl mapping are saved in `assets/replogle_scldm2k_gene_order.*.txt`.
The original loader filters each native gene panel, normalizes by that file's
median full-panel library size, applies log1p, and then selects the ordered panel.
Raw-data audit: 966,728 cells, 938,551 train and 28,177 held out
(HepG2 × the published 372 targets). All 2,000 genes are present in every file.

```bash
# On the login node, using the project's conda environment:
python scripts/data/prepare_replogle_stage1.py
python scripts/checks/audit_replogle_stage1_raw.py
bash scripts/launch/submit_replogle_stage1.sh --dry-run
# Submit once; an existing job with the same name is never overwritten:
bash scripts/launch/submit_replogle_stage1.sh --submit
rjob get scevae-wangyue-replogle-stage1-20261117
```

The durable rjob uses one GPU in `beam_gpu_pool`, the published image pinned by
digest, and mounts both the project and the source data. It runs GPU environment
checks, the original eager loader and split checks, a separate 100-step dry run,
and frozen-checkpoint token/decode checks before starting the formal run. The
existing training loop includes step zero, so its dry run performs 101 updates.
Full eager loading requires the worker's memory; do not run it on the login node.
Source/config hashes are checked when the queued worker starts.

The first baseline uses MSE reconstruction, 64 FSQ tokens with 512 codes per
position (levels 8×8×8), seed 401, batch 128, bf16, 75,000 iterations,
AdamW lr 6e-4, weight decay 1e-3, 2,500 warmup steps and cosine decay. Encoder and
decoder are trained together without perturbation/cell-type conditioning or
auxiliary classifiers. The smoke run has its own weights, optimizer and sampled
statistics; only its cached DEG table is reused for the formal run's metrics.
The existing code constructs this metric cache on the full dataset. The published
test slice is used as validation by the original recipe; keep the fixed final
training endpoint rather than selecting hyperparameters on this slice.

Logs and reports are in `artifacts/replogle_stage1/` (`worker_phase.tsv`,
`worker.log`, `raw_audit.json`, `loader_audit.json`, checkpoint acceptance JSON).
W&B records are written offline on GPFS. Runs are saved separately:

```text
outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401-smoke/
outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401/
```

Each original training run saves config, dataset statistics and Accelerate
model/optimizer/scheduler/RNG checkpoints. A freshly computed DEG table is saved
locally; a run using `dataset.deg_table_path` keeps using that shared cache.
Resubmission with identical
sources/config resumes through the original `checkpoints/latest` pointer. After
completion, Stage 2 can use the formal directory as `model.vae_path`; acceptance
checks use the same `load_vae_from_run` factory to verify freezing and token IDs
in `[0, 511]`. GPU checks and training begin only after the queued job is scheduled.

The 2026-11-17 run completed all 75,000 steps on one NVIDIA H200 in
`beam_gpu_pool`. Formal phase: 2026-11-17 21:55:56 to 2026-11-18 00:36:58 HKT
(about 2 h 41 min including loading, evaluation and checkpointing). Smoke profiling
measured 8.10 steps/s and 16,074 MiB peak allocated GPU memory at batch 128.
Final held-out reconstruction MSE was 0.1643; the final checkpoint is
`outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401/checkpoints/step_0075000_final`.
The scheduler reports this historical rjob as Failed because the original final
acceptance incorrectly required a local DEG-cache copy. The validator now follows
the saved config's cache path. A separate recheck receipt records final checkpoint
acceptance without rerunning training or changing the historical job status.
The final checkpoint passed a CPU fp32 recheck on 128 real cells (64 train,
64 held out): frozen factory loading, finite weights/reconstruction, 64 integer
token IDs per cell within `[0, 511]`, and exact token-to-code round trips.
That sample used 499 distinct codes. The GPU smoke check had already passed.
The completion receipt is `artifacts/replogle_stage1/completion.json`.

```bash
# Lightweight checkpoint recheck on CPU, one cell at a time, without gradients:
python scripts/checks/validate_replogle_stage1.py \
    --checkpoint-run outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401 \
    --device cpu
```

Training is two-stage here.

**Stage 1 — train the `ExpressionVAE`** (Replogle example):

```bash
uv run accelerate launch scripts/train.py \
    --config_path config/final/vaes/reploall-vae-fsq-mse.yaml
```

This writes an experiment directory under `outputs/` containing `config.yaml`,
`dataset_stats.json`, and `checkpoints/`.

**Stage 2 — freeze the VAE and train a prior** over its code sequence:

The local AR baseline is prepared in
`config/local/replogle-stage2-ar-fsq-mse.yaml`, derived directly from
`config/prior/reploall-autoregressive-fsq-mse.yaml`: seed 420, 75,000 steps,
lr 2e-4, warmup 3,000, clip 1.0, batch 128, six DiT layers of width 384.
It references the completed local Stage 1 VAE and the same raw-file loader,
gene order, normalization, and scLDM holdout. Targets are encoded online by
the existing `AutoregressiveModel`; the original `scripts/train.py` and
`scripts/eval.py` remain the training and real ancestral-sampling entry points.

One small model fix keeps the frozen VAE in eval mode when the AR parent is
switched to train mode, so tokenizer dropout is not reactivated. The worker
imports the checkout through a package-name alias matching the wheel mapping.
CPU preflight passed: finite forward/backward and optimizer updates, unchanged
VAE weights, deterministic target tokens, and condition-only generation of
64 tokens followed by decoding to 2,000 expression values.

There is an unresolved source discrepancy: Appendix A of the paper names
ESM2-3B; Meta's 3B config has hidden size 2,560. The repository instead points
to Arc's `competition_support_set/ESM2_pert_features.pt`, whose downloaded
vectors have 5,120 entries. This is not a direct numeric quote of 2,560 from
the paper, and feature width alone does not prove the originating model.
The user selected the config's 5,120-dimensional feature source on 2026-11-18;
the selection is recorded in `artifacts/replogle_stage2/feature_choice.json`.

The Arc ZIP member was extracted by byte ranges and CRC-checked; original
vectors are preserved. `scripts/data/align_replogle_esm_symbols.py` uses a hashed
HGNC snapshot to align 49 historical labels without changing vectors. All
372 held-out targets then have nonzero features. Five training labels remain
unresolved, plus `non-targeting`; the original loader uses zero vectors for
those labels. Reports are in `artifacts/replogle_stage2/`.

```bash
python scripts/data/prepare_replogle_stage2.py
# HGNC snapshot: artifacts/replogle_stage2/hgnc_complete_set.txt
python scripts/data/align_replogle_esm_symbols.py
PYTHONPATH="$PWD/artifacts/replogle_stage2/source_overlay" \
    python scripts/checks/validate_replogle_stage2.py
bash scripts/launch/submit_replogle_stage2.sh --dry-run
# Submit the selected 5120-dimensional AR baseline:
bash scripts/launch/submit_replogle_stage2.sh --submit
```

```bash
uv run accelerate launch scripts/train.py \
    --config_path config/final/prior/reploall-autoregressive-fsq-mse.yaml
```

The prior config references the trained VAE via `model.vae_path:`. Point it at
your Stage-1 run directory (set `prefix_run_name_with_datetime: false` on the VAE
config, or edit `vae_path` to match the actual run-dir name).

### Parameterized Replogle experiments

`scripts/launch/run_replogle_experiment.sh` currently supports the author's
AR–FSQ–MSE and AR–FSQ–NB combinations. It renders configs from the existing
presets and calls the original `scripts/train.py` / `scripts/eval.py`.

Use separate entry points when training the two stages independently:

```bash
# Stage 1 only: train/check the NB tokenizer. Other hyperparameters have defaults.
bash scripts/launch/run_replogle_stage1.sh --head nb --submit

# Stage 2 only: use a finished Stage 1 run (replace the example path).
bash scripts/launch/run_replogle_stage2.sh \
    --vae-path /path/to/completed/stage1 --submit

# With the existing completed MSE tokenizer, preview the Stage 2 job:
bash scripts/launch/run_replogle_stage2.sh \
    --vae-path outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401 --dry-run
```

Stage 1 runs its smoke/formal training and checkpoint acceptance, then exits.
It does not require ESM features or construct/train an AR model. Stage 2 reads
the supplied tokenizer's head, token count and codebook size by default, verifies
its config/statistics/final checkpoint, and runs only AR smoke/formal training,
checkpoint acceptance and real ancestral-sampling evaluation. Explicit mismatched
head/token/vocabulary options fail before submission. The source VAE is frozen
and its checkpoint digest participates in the new experiment ID.

The shared launcher also accepts `--stage all|stage1|stage2`. Its default is
`all`; for compatibility, supplying `--vae-path` without `--stage` selects
`stage2`. Explicit `--stage stage2` requires `--vae-path`.

```bash
bash scripts/launch/run_replogle_experiment.sh \
    --prior ar --quantizer fsq --head nb \
    --batch-size 128 --num-latent-tokens 64 --codebook-size 512 \
    --stage1-seed 401 --stage2-seed 420 \
    --stage1-steps 75000 --stage2-steps 75000 \
    --stage1-lr 6e-4 --stage2-lr 2e-4 \
    --submit
```

Use `--dry-run` to render configs and print the launch command, `--spec` to
render rjob YAML, `--preflight` for a bounded CPU check, and `--help` for all
hyperparameters (warmup, clipping, weight decay, AR width/depth, FSQ levels,
NB temperature and dispersion floor). Submission automatically runs CPU
preflight if needed. GPU allocation is currently single-card in `beam_gpu_pool`.

NB uses raw integer UMI (`apply_normalize: false`, `apply_log1p: false`) and
requires its own jointly trained FSQ encoder/NB decoder. Its Stage 2 still
uses next-token cross entropy; NB is the frozen VAE's expression head.
Each experiment has an isolated output directory and its own Stage 1 DEG
cache, so a raw-count NB run never reuses a log1p-MSE metric cache.
The worker runs data checks, Stage 1 smoke/formal training, checkpoint checks,
Stage 2 smoke/formal training, and real ancestral-sampling evaluation.

To reuse the completed MSE tokenizer for a separate AR experiment:

```bash
bash scripts/launch/run_replogle_experiment.sh --head mse \
    --vae-path outputs/replogle_stage1/replogle-stage1-fsq-mse-seed401 \
    --dry-run
```

The reused VAE must match the requested head, token count and vocabulary;
incompatible combinations fail before submission. FSQ mappings are checked
against the actual vocabulary size (the author's automatic mapping for 1024
produces 1000 codes; use explicit `--fsq-levels` if an exact product is needed).

Resolved configs are saved under `config/local/replogle_experiments/<id>/`,
parameters, input/source hashes, preflight and worker logs under
`artifacts/replogle_experiments/<id>/`, and checkpoints/evaluation under
`outputs/replogle_experiments/<id>/`. The hyperparameter hash in the ID
prevents different settings from sharing optimizer state or checkpoints.
An optional `--tag` creates a separately named repeat.

The two standalone entry points were checked with CPU probes: Stage 1 updates
only a fresh NB tokenizer; Stage 2 loads the existing MSE tokenizer and updates
only the AR model while keeping the VAE weights unchanged. Previously queued
schema-1 NB jobs use the frozen `scripts/lib/replogle_experiment_v1.py` worker and
their original execution plans. New CLI functionality does not alter those plans.

The original queued NB rjob `scevae-wangyue-ar-fsq-nb-990fc9c87a` was stopped
and replaced on 2026-11-18 by `scevae-wangyue-ar-fsq-nb-2g-0868-20261118`:
two H200 GPUs in `beam_gpu`, pinned to
`gpu-lg-cmc-h-h200-0868.host.h.pjlab.org.cn`. Its isolated experiment ID is
`ar-fsq-nb-l64-k512-990fc9c87a-2gpu-0868`; the artifact directory records the
submission command and configuration. Each rank uses batch 64 (global 128).
Both stages still train for 75,000 optimizer updates, with one scheduler step
per update. The artifact-local training entry copies the original trainer with
only scheduler stepping and rank-zero profiling adjustments; the shared trainer
and the separately queued MSE job's pinned sources remain unchanged.

The agreed text conditions can be generated on CPU, without reading expression
matrices or loading any model:

```bash
python scripts/data/prepare_replogle_text_conditions.py
```

`artifacts/replogle_text_conditions/v1/conditions.jsonl` contains one record per
observed cell-line/perturbation combination; `conditions.csv` is the equivalent
table for inspection. The four-line template is versioned in `template.json`:

```text
Task: Single-cell Perturbation Prediction
Cell Line: HepG2
Perturbation Type: CRISPRi
Target Gene: EIF2S1
```

For existing `non-targeting` samples, the last two values are `Non-targeting
control` and `none`. Cell-line names are K562, HepG2, Jurkat, and RPE1; target
gene symbols preserve the existing training labels. The first generated table
contains 9,240 conditions (8,868 train, 372 heldout), covering all 966,728 retained
cells, including 39,165 non-targeting cells. Counts and split labels are metadata
outside the condition text. No expression values or response summaries enter the
text, including for heldout conditions.

`cell_condition_index.npz` maps each retained cell to a zero-based condition row
using its source-file index and physical H5AD row. `manifest.json` documents that
index, input fingerprints, coverage checks, and output hashes. `examples.json`
shows HepG2–EIF2S1 (78 cells) and HepG2 non-targeting (4,976 cells). These files
prepare text conditions for future work; no text embeddings are computed and
the existing AR training inputs are not changed.

The separate text-embedding route was cancelled. Its one-off encoding scripts have been retired; the condition text preparation above remains in use.

The single-cell LLM prior uses **Qwen3.5-0.8B full language-model SFT with
the frozen NB-FSQ tokenizer**, without a separate text embedding model or CPT.
The original text-embedding jobs were cancelled. LLM input is the approved
four-line condition text and the current generated cell's code prefix; it does
not contain observed control-cell codes. Each independent completion has 64 FSQ
codes, a cell-end marker and chat EOS. Generated cells are accumulated by
condition for the original population evaluation.

`scripts/data/prepare_replogle_llm.py` fixes 200 HepG2 training conditions as internal
validation before fitting the LLM, stratified by group size among groups with at
least 31 cells. The seed-420 split has 924,460 SFT training cells, 14,091 validation
cells and the unchanged 28,177 official test cells (372 conditions). The same
genes in other cell lines remain train. This is validation of Stage 2 with a
fixed tokenizer; that tokenizer was previously fitted on the original train
split, including cells now reserved for internal LLM validation.

The experiment is recorded beneath
`artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1/`. Its `split.json` fixes the
condition lists; `cell_fold.npy` aligns with the retained cells in loader order.
`cache/token_ids.npy` contains all 966,728 NB-FSQ sequences as uint16 and
`cache/condition_rows.npy` maps them to the existing condition text table.
The source NB checkpoint is `step_0075000_final` with SHA256
`a020be5ba25d7c6edb59f8fbfef3f648866ed9fde0128b6fe2d83952400cc25d`.

The local Qwen3.5-0.8B language weights are exported into an independent
directory; the original model is unchanged. Its visual encoder and MTP modules
are excluded, all language parameters are trained, and 512 FSQ tokens plus two
cell-boundary tokens are added. CPU checks verify that the full condition text
survives chat formatting, each FSQ code is one token, and completion-only loss
with cropped prompt logits equals full-sequence loss. Actual sequence lengths
are 106–115 tokens. Separate pinned Transformers/FLA dependencies leave the
original experiment environment unchanged.

```bash
# CPU checks only; no GPU job submitted.
bash scripts/launch/submit_replogle_llm.sh --check
# Submit the gated dual-GPU training pipeline.
bash scripts/launch/submit_replogle_llm.sh --submit
```

The pipeline requests two GPUs and sixteen CPU cores in `beam_gpu_pool`, with
no fixed node. The public pool determines the final CPU and memory allocation
from its GPU SKU; training uses eight CPU threads per rank. It first runs 50 smoke updates and
condition-only generation/NB decoding checks, then two 1,000-update learning-rate
pilots (1e-5 and 3e-5). Selection uses population Pearson delta on a fixed subset
of 20 internal validation conditions, with token loss breaking ties. Formal
training resets to the original language weights and runs three epochs, batch
8 per GPU with accumulation 8 (global batch 128), using only the new train split.
Checkpoint selection uses internal validation loss; final population validation
and official test evaluation follow training. Progress is saved to
`worker_phase.json` and per-run `metrics.jsonl` beneath
`outputs/replogle_llm/qwen35-08b-ar-fsq-nb-v1/`.

`scripts/evaluation/eval_replogle_qwen.py --protocol legacy-text` replaces only model loading and sampling in
`scripts/eval.py`; preprocessing and population metrics are reused unchanged.
Comparative NB evaluation uses the same observed-HVG-depth decoding convention
as the original AR baseline, recorded explicitly in the evaluation provenance.
The generation smoke check instead uses the stored training library-size prior
and does not read target expression or target codes. PDS target-gene exclusion
is checked using the existing symbol-alignment audit.

### Independent control-cell A/B training

The new experiment `qwen35-08b-fsq-nb-hepg2-v1` lives under
`artifacts/replogle_llm_control/`, `config/local/replogle_llm_control/` and
`outputs/replogle_llm_control/`. Variant **A** takes perturbation text; variant
**B** adds one non-targeting cell's 64 FSQ codes. Both predict one cell's 64 codes
and use the original Qwen3.5-0.8B language-only export, never the previous SFT
checkpoint. The existing NB cache/tokenizer and environment are read-only inputs.
Existing `qwen35-08b-ar-fsq-nb-v1` scripts, configurations and results are preserved;
output guards reject paths outside the new experiment and reject reused run IDs.

K562, Jurkat and RPE1 supply the source data. Whole cell-line/target combinations
are withheld, stratified by cell line and group size, retaining every held-out
gene in another training cell line. The fixed split contains:

| Split | Cells | Conditions | Role |
|---|---:|---:|---|
| Train | 743,078 | 6,208 | Includes 34,189 source-line non-targeting cells |
| Combination validation | 39,123 | 318 | Known gene and line, new combination |
| Combination test | 39,054 | 320 | Known gene and line, new combination |
| HepG2 test | 28,177 | 372 | Known gene, cell line unseen during SFT |

The validation loss monitor fixes 20 cells per condition (6,360 total).
HepG2 contributes no SFT training cells. Its 4,976 controls are partitioned per
`gem_group` into 2,474 inference-support and 2,502 metric-reference cells; the
remaining 112,320 HepG2 perturbed cells are unused. The frozen NB tokenizer
previously saw HepG2, so this is **Stage-2** cell-line generalization.

B draws controls from the same source cell line and `gem_group`, resampling each
epoch with deterministic seeds; validation controls are fixed. Non-targeting
targets cannot draw themselves as controls. These are unpaired reference cells,
not observed before/after pairs. Only the completion (64 FSQ codes, cell-end,
EOS) contributes to loss. Both variants use the same target split, sampling
order, base-model initialization, learning rate 3e-5, three epochs and global
batch 128. With per-GPU batch 8, gradient accumulation is 8 on two GPUs and 4 on
four GPUs. No learning-rate scaling is applied. Total CPU threads are 32.

```bash
# Print commands only: no jobs submitted. Unified defaults: 8 GPUs, beam_gpu, no fixed node.
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-a
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-b

# Verify CPU acceptance, frozen inputs and baseline preservation; no submission.
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-a --check

# When GPU execution is requested, submit A and B separately.
# Each unified train job first gates formal training on 10 smoke updates + free generation.
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-a --gpus 4 --per-device-batch-size 8 --stage train --submit
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-b --gpus 4 --per-device-batch-size 8 --stage train --submit

# Optional scaling measurements, each with a distinct output directory.
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-b --gpus 2 --per-device-batch-size 8 --stage benchmark --submit
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-b --gpus 4 --per-device-batch-size 8 --stage benchmark --submit
```

`benchmark` performs the smoke gate, then 200 full-data optimizer updates and
records throughput over steps 51–200, excluding startup/JIT. Compare the same
variant and GPU type at fixed global batch, using `throughput.json`.
`--stage smoke` only performs the smoke gate. `--run-id` selects a fresh run
directory; reusing an existing one fails without overwriting anything.
The current requested placement is four GPUs per experiment in `beam_gpu`,
with A on `gpu-lg-cmc-h-h200-0868.host.h.pjlab.org.cn` and B on
`gpu-lg-cmc-h-h200-0862.host.h.pjlab.org.cn`. Submission reads each variant's
quota group and node from its JSON config, and training verifies both at runtime.
The published image supplies PyTorch; the validated Transformers/FLA/Triton
packages are imported from the prior experiment. Runtime caches and logs go to
the new run. Each job requests 32 CPU cores and 96 GiB of host memory.

Preparation is CPU-only via `scripts/data/prepare_replogle_control_ab.py`; checks via
`scripts/checks/check_replogle_control_ab.py` require the existing isolated package
directory on `PYTHONPATH`. Checks validate split integrity, control sampling,
target-free prompts, output-only loss, tiny-model backward/optimizer updates,
2/4-GPU batch arithmetic and preservation of the baseline files. The generated
`ready_inputs.json` pins all training inputs; changed files require rechecking.
Actual multi-GPU training and generation remain to be checked by the smoke gate.

These scripts stop after training; population evaluation is not automatically
launched. `evaluation_plan.json` records the three evaluation splits. NB decoding
depth (control depth or a source-training prior) remains to be chosen consistently
for A/B before population evaluation; it does not affect SFT on cached tokens.

### Independent control-set C training

C takes the same perturbation text plus **N non-targeting control cells**, each
with 64 FSQ tokens, and predicts **one** cell's 64 FSQ tokens. Repeated stochastic
single-cell completions can later be pooled for population evaluation. N is an
input reference-set size, not the number of output cells or the training batch.
The command-line parameter is **`--control-cells-nums`**, default **8**.
Each reference is serialized as `<cell_start>` + 64 codes + `<cell_end>` under
`Control Cells:`. Only the output completion contributes to loss.

C reuses the exact A/B split and frozen NB cache, initializes from the original
Qwen language export, and retains the same target cells, learning rate, epochs,
and global batch 128. Separate C scripts/configs/artifacts/results live under
`qwen35-08b-fsq-nb-hepg2-c-v1`; neither running A/B scripts nor their pinned inputs
are changed. Source controls are drawn within the same cell line and `gem_group`,
excluding the target itself. Sampling is without replacement when enough controls
exist; otherwise all eligible controls are included and the remaining slots are
filled with replacement, then shuffled. Training resamples each epoch; validation
is fixed. HepG2 controls use only the A/B inference-support pool, never the
metric-reference pool. These controls are unpaired references, not measured
before/after counterparts. This sequential serialization does not impose
permutation invariance on the model.

```bash
# Print the command only; these examples DO NOT submit jobs.
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-c --control-cells-nums 8
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-c --control-cells-nums 16
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-c --control-cells-nums 32 --per-device-batch-size 1

# Verify prepared inputs for the requested N; no submission.
bash scripts/launch/submit_replogle_qwen.sh --preset hepg2-c --control-cells-nums 8 --check

# CPU preparation/acceptance, using the same isolated environment as A/B.
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
PYTHONPATH="$PWD/artifacts/replogle_llm/qwen35-08b-ar-fsq-nb-v1/environment/python_packages:$PWD/scripts" \
/mnt/shared-storage-user/wangyue/conda/envs/sc-evae/bin/python scripts/checks/check_replogle_control_c.py
```

Only an explicit `--submit` submits an rjob. Defaults are four GPUs, 32 CPUs,
96 GiB host memory and `beam_gpu`, with no fixed node for C; `--node HOST` can
specify placement later. The default per-GPU train/eval batch is 2, giving 16
accumulation steps on four GPUs. `--per-device-batch-size` adjusts memory use while
keeping global batch 128. Prompt capacity grows with N and is checked against the
model context limit; cells are never silently truncated. GPU memory/speed for C
have not yet been measured. Every future train job first runs 50 smoke updates
and a generation check before starting formal training from pretrained weights.

Run paths include N: `outputs/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-c-v1/n8/<run-id>/`.
The resolved N, microbatch, accumulation and actual maximum sequence length are
saved in each phase's `run_config.json`; resumed training must retain them.
`--run-id` must be fresh for a new run. Like A/B, C stops after training; the
nine-metric population evaluation and a consistent NB decoding-depth rule remain
separate follow-up work. CPU acceptance checks N=1, 8 and 32 (plus a requested N),
including support-pool shortages, self-exclusion, target-free inputs, deterministic
validation and the long-prompt completion loss. The existing A/B input manifest
is verified without being rewritten.

### Historical C continuation

C completed its one-off checkpoint-6000 resource migration on four GPUs. The fixed-job handoff/continuation launchers have been retired. Checkpoints, migration records, curves and completed evaluations remain available; see `docs/replogle_control_ab_stage2_results.md` in the original experiment workspace (not included in this code copy).

### A/B population evaluation: observed target depth and full HepG2

The approved evaluation protocol keeps the author's NB sampling path:
`vae.decode_observed(codes, None, target_counts.sum(-1, keepdim=True))`.
No Stage-1 weights or depth prior are changed. Target expression and target FSQ
codes never enter the LLM; the target HVG count total is used only by NB decoding.
A/B use the actual exported step-11,000 models, not the unsaved step-11,500 minima.

Four requested sets are reported: combination validation (318 conditions, 39,123
cells), combination test (320, 39,054), full held-out HepG2 (2,393, 140,497), and
its official paper subset (372, 28,177). Every target cell receives one independent
64-code draw. The paper subset reuses the exact same cell predictions from the
full HepG2 generation cache. The source combination sets are evaluated separately
for K562/Jurkat/RPE1; a clearly labelled unweighted mean of their three cell-line
scores is also saved. DE controls and PDS candidate sets are never mixed across
cell lines. Raw per-line metrics and their valid counts remain available.

Input controls retain the frozen same-line/gem support pools. Source metric
reference controls are deterministically capped at 5,000 per line. HepG2 uses
its frozen **2,502-cell metric-reference pool**, disjoint from the 2,474 input
support controls; this differs from the older baseline's larger control pool and
must be disclosed when comparing scores. The NB depth and metric definitions
remain the original ones. Gene-symbol target exclusion is audited with the
existing helper. All nine requested metrics are reported: W2, multi-scale MMD2,
FD, PDS, PR-AUC, Sp-sig, Sp-LFC, ovl@N and P-delta. PCA-30 and the existing
5,000-cell distribution-metric cap are retained; no target group is truncated.

`eval_replogle_qwen.py --protocol legacy-control --variant A|B|C` changes only dataset/prediction adapters and invokes
`scripts/eval.py` unchanged. It saves raw generated counts/codes and true counts
in an HDF5 cache, then computes CPU metrics from that cache. CPU acceptance
checks frozen splits and runs an identity-prediction fixture through the actual
original evaluator and symbol-alignment audit; this fixture is not an experiment.

```bash
# Dry runs only by default. Explicit --submit creates the requested rjob.
python3 scripts/launch/submit_replogle_control_eval.py A --submit
python3 scripts/launch/submit_replogle_control_eval.py B --submit
```

Each job requests one GPU, 16 CPUs and 96 GiB memory, in **beam_gpu** on
**gpu-lg-cmc-h-h200-0863.host.h.pjlab.org.cn**. Inference batch defaults to 64.
The independent outputs live in
`outputs/replogle_llm_control_evaluation/ab-observed-depth-v1/{A,B}/<run-id>/`;
plan, checked input hashes and job receipts live in
`artifacts/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-v1/evaluation_observed_depth_v1/`.
These paths do not overwrite previous training or baseline evaluation results.

### Config layout (`config/final/`)

- `vaes/`     — Stage-1 `ExpressionVAE` configs. `{dataset}-vae-{fsq|gaus}-{head}.yaml`
- `prior/`    — Stage-2 priors. `{dataset}-{autoregressive|mdlm|flow}-{fsq|gaus}-{head}.yaml`
- `nolatent/` — ablation priors trained directly on counts (no VAE latent)

Datasets are `reploall` (Replogle) and `parse1m`. Heads are `mse`, `nb`,
`hurdle`, `ce-quantile`. Bottlenecks are `fsq` (discrete) and `gaus` (Gaussian,
used with flow and in the TeloHAEC grid).

## Evaluation

**Per-perturbation + distribution metrics** (reads `{train_dir}/config.yaml`):

```bash
uv run python scripts/eval.py --train_dir outputs/<your_prior_run>
```

This computes three metric families, written to
`{train_dir}/eval_results/summary_metrics.json`:

- `distribution` — W₂, MMD², Fréchet distance (`metrics/distribution.py`)
- `cell_eval`    — disc_l1, PR-AUC, Spearman-sig/LFC, overlap@N, Pearson-Δ
(`metrics/state_metrics.py`) — **the paper's reported per-perturbation suite**

**TeloHAEC reversion benchmark** (frozen encoder):

```bash
uv run python scripts/eval_pfizer.py \
    --train_dir outputs/<your_vae_run> \
    --h5ad_path datasets/pfizer/pfizer_raw.h5ad \
    --gene_order_path assets/replogle_scldm2k_gene_order.txt --num_hvg 2000
```

The raw file is sliced to the model's 2000-gene panel on the fly by the
data loader (`gene_order_path` permutes + zero-fills; `num_hvg` slices) — no
separate alignment step. Reports per-perturbation Calinski-Harabasz separability
and the NF-κB enrichment AUC (`metrics/pfizer_ranking.py`).

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{Bhattacharya2026,
    author    = {Bhattacharya, Sanjukta and Gensbigler, Christian and Karim, Shaamil and Lees, Jon},
    title     = {Elucidating the Design Space of Generative Models for Single-Cell Perturbation Prediction},
    year      = {2026},
    doi       = {10.64898/2026.06.15.732063},
    publisher = {Cold Spring Harbor Laboratory},
    journal   = {bioRxiv},
    url       = {https://www.biorxiv.org/content/10.64898/2026.06.15.732063v1}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.

## Acknowledgements

This repository builds heavily off of [DCM](https://github.com/sanjukta7/aivc-dcm), [STATE](https://github.com/ArcInstitute/state),
[scLDM](https://github.com/czi-ai/scLDM), [DiT](https://github.com/facebookresearch/DiT),
and [Score-Entropy Discrete Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).
We also used the [cell-load](https://github.com/ArcInstitute/cell-load) package introduced in the STATE repository. Here's the phenotypic reversion benchmarl: [teloHAEC, phenotypic reversion](https://github.com/pfizer-opensource/phenotype_reversion).

## Dataset Download

We use three benchmarks. Download each dataset and place it under `datasets/`:


| Dataset    | Description                                                              | Link                                                                                                                                                                                                     |
| ---------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `replogle` | Genome-scale Perturb-seq (Replogle et al. 2022)                          | [Figshare](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387) |
| `pbmc-1m`  | 1M PBMC cytokine perturbation (Parse)                                    | [Figshare](https://figshare.com/articles/dataset/pbmc_parse/28589774?file=53372768)                                                                                                                      |
| `teloHAEC` | Phenotypic reversion and target prioritization for cellular inflammation | [Zenodo](https://zenodo.org/records/18792213)                                                                                                                                                            |
