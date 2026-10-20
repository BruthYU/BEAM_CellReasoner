# CellTok data inventory

Audited against CellTok bioRxiv v1 (doi: `10.1101/2025.10.22.684047`) on
2026-10-20.

## What is present

- `5000_gene_names.csv`: the 5,000-gene alignment vocabulary used by the cell
  tokenizer.
- `blood/`: the repository's 4,755-cell tutorial example, including the source
  H5AD, 32-code representation, complete annotation and generation Arrow
  shards, and a 100-cell generated result.
- `paper_evaluation/`: publicly identifiable source datasets used in the paper,
  fetched by `scripts/fetch_paper_data.sh`. `SHA256SUMS` records the downloaded
  bytes. It currently contains 37 verified source files (4.968 GiB): 430,903
  small-intestine cells, 106,136 brain cells, 126,564 blood/airway cells,
  116,313 GSE171524 lung cells, and the GSE135893 source matrix. These are
  source datasets, not the unpublished CellTok train/test splits or tokenized
  derivatives.

## Paper stages and reproducibility status

| Stage | Paper data | Local status |
| --- | --- | --- |
| Cell tokenizer training | 68M filtered CELLxGENE cells; 63.8M train / 4.7M test by study | **Not reproducible exactly.** The paper does not publish the study list, frozen CELLxGENE snapshot, filtering pipeline, or split manifest. |
| CellTok-base training | 5M cell-code/text pairs plus an unnamed Hugging Face general corpus (1:1) | **Missing and not reproducible exactly.** GPT-4o-generated descriptions, sample IDs, and the text-corpus ID are not published. |
| Tokenizer evaluation | GSE135893 and GSE171524 | Public source data is under `paper_evaluation/`. |
| Cell annotation/generation | small intestine (Kong 2023), brain (Leng 2021), blood (Szabo 2021); paper uses 10k train + 1k test per dataset | Public source data is under `paper_evaluation/`; the paper's exact sampled splits are not published. |
| Population annotation and disease state | GSE171524 with named donor splits | Public source H5AD is under `paper_evaluation/GSE171524/`; derived groups and tokenized prompts still need construction. |
| Cell-cell communication | GSE171524, CellChat labels, 73 pathways | Source H5AD is present; CellChat outputs and sampled source-target groups are not published. |
| Tutorial-only spatial CCC | Visium HD colorectal-cancer inputs referenced by tutorial 8 | Missing; the notebook contains private paths and no public accession. |
| Tutorial-only trajectory | `trajectory_adata.h5ad` referenced by tutorial 9 | Missing; the notebook gives no source accession. |

The CELLxGENE downloads are pinned to the dataset-version URLs available on
2026-10-20. They are the closest authoritative public sources, but cannot be
claimed byte-identical to the unreleased 2025 training/evaluation snapshot.
The GSE135893 matrix has 33,694 genes and 220,213 barcode columns; its supplied
IPF metadata contains 114,396 QC/annotated cells, consistent with the source
study's reported analyzed subset.

Authoritative source records:

- CellTok paper: <https://doi.org/10.1101/2025.10.22.684047>
- Small intestine: <https://cellxgene.cziscience.com/collections/5c868b6f-62c5-4532-9d7f-a346ad4b50a7>
- Brain: <https://cellxgene.cziscience.com/collections/180bff9c-c8a5-4539-b13b-ddbc00d643e6>
- Blood: <https://cellxgene.cziscience.com/collections/29f92179-ca10-4309-a32b-d383d80347c1>
- GSE171524: <https://cellxgene.cziscience.com/collections/e4c9ed14-e560-4900-a3bf-b0f8d2ce6a10>
- GSE135893: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135893>

## Validation

Run:

```bash
bash scripts/validate_paper_data.sh
```

To resume downloads on this GPFS cluster through its faster local route, use:

```bash
CELLTOK_PROXY_URL=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 \
  bash scripts/fetch_paper_data.sh
```

The validator checks expected files and sizes, gzip integrity, HDF5 readability,
AnnData dimensions, all Arrow shard references, and SHA-256 checksums.
