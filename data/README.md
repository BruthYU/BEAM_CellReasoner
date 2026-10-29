# Stage1数据接口

本PR只提交 `5000_gene_names.csv`，不提交表达矩阵、训练清单、权重或个人路径。
词表的顺序就是模型的5000维输入顺序。源H5AD通过分块读取转换成稀疏counts归档，
loader按batch读取并执行 `log1p(10000 * counts / aligned_5000_gene_sum)`。
零表达行保持0；缺失基因补0，重复源基因按映射累加。归一化分母是对齐后的5000基因。

## 本地数据位置

现有共享实验目录下的数据为：

```text
data/stage1/full_primary/
  fit_full_primary_20261026.csv
  cellstream/
  cellstream_large_shards/
```

正式清单包含473个train归档（54,976,407 cells）和18个test归档
（3,389,742 cells），train/test的study不重叠；另有36个benchmark归档不纳入训练。
不要直接glob存储目录：这会混入隔离数据或遗漏分片。Non-primary增量未并入该训练协议。

这些数字描述本地冻结协议，不是作者精确63.8M/4.7M划分。数据需由用户自行准备；
脚本默认的manifest路径只是约定，并不表示仓库中包含这些数据。

## 新数据转换与清单

从仓库根目录执行：

```bash
TASK=convert INPUT=/path/source.h5ad OUTPUT=/path/study.csad \
  SESSION=celltok-convert bash scripts/data.sh
```

默认筛选 `obs.is_primary_data=True`，缺少该字段则保留全部行，并记录来源证据。
`raw-or-X` 优先取raw，否则取X；只有有限、非负、整数counts才能写入counts归档，
不能把已经log归一化的X当作counts。转换器还记录词表SHA和筛选/存储方式，供训练前审计。
`CELL_SELECTION=non-primary` 或 `all` 可生成其他数据，但本次训练入口要求primary来源。

CSV示例（归档路径相对CSV所在目录）：

```csv
dataset_id,study_id,split,selected_cell_count,cellstream_path
study-a,study-a,train,12000,cellstream/study-a.csad
study-b,study-b,test,3000,cellstream/study-b.csad
```

按study划分，而不是在同一study内随机分行。`selected_cell_count`须等于该归档的行数；
归档需由本转换器生成，或具有等价的 `uns['celltok_stage1']` 来源、词表SHA及primary证据。

大H5AD可按筛选后的行范围转换为多个归档；直接调用同一转换器：

```bash
PYTHONPATH=src:. python -m celltok data convert \
  --input /path/source.h5ad --output /path/shard0.csad \
  --matrix raw-or-X --storage counts --block-rows 8192 \
  --selected-cell-start 0 --selected-cell-stop 100000
```

`--selected-cell-start/stop`是筛选后细胞序列的左闭右开范围。修改范围与输出路径即可处理下一片。
`data freeze`可从原清单与分片清单生成只包含完整数据集的新清单，校验连续覆盖和study隔离，
不会覆盖已冻结清单。分片清单保留 `dataset_id/study_id/split/cell_selection`，并记录
`selected_cell_start/selected_cell_stop/selected_cell_count/source_dataset_selected_cell_count`。

```bash
TASK=freeze MANIFEST=/path/source.csv SHARD_MANIFEST=/path/shards.csv \
  OUTPUT=/path/complete.csv bash scripts/data.sh
TASK=validate MANIFEST=/path/complete.csv OUTPUT=outputs/data/validation.json bash scripts/data.sh
```

## 读取与缓存

```python
from lib_vqvae.cellstream_dataset import CellStreamDataset

dataset = CellStreamDataset.from_manifest(
    "data/stage1/full_primary/fit_full_primary_20261026.csv",
    split="train", expected_genes=5000,
    target_sum=10000, log_transform="log1p",
    local_cache_dir="/tmp/celltok-training", read_threads=1,
    max_open_archives=1,
)
batch = dataset.get_batch_numpy([0, 1])  # [2, 5000], float32
dataset.close()
```

CellStream packed reader需要支持mmap的文件系统。本地集群的共享挂载需先复制归档至节点本地缓存，
全量train/test约需55.56 GB缓存空间；缓存会核对容量并复用已有文件。
训练前审计全部归档的shape、词表顺序、counts和primary证据，并检查train/test study隔离。

历史冻结CSV内的三个旧Stage1根路径由 `lib_vqvae.paths.resolve_data_path` 映射到新目录；
不需要建立旧目录软链接。新数据建议直接写规范路径。
