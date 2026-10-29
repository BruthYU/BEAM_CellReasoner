"""Shared loader preparation; training loops remain in lib_vqvae.VQVAE."""
import csv
import hashlib
from itertools import islice
from pathlib import Path

import numpy as np
from tqdm import tqdm

class LimitedLoader:
    """Bound only smoke-test data iteration; model.fit still owns all training."""

    def __init__(self, loader, steps):
        self.loader = loader
        self.steps = min(len(loader), steps)
        self.batch_sampler = loader.batch_sampler

    def __len__(self):
        return self.steps

    def __iter__(self):
        return islice(self.loader, self.steps)


def audit_dataset(dataset, manifest, split, genes_path):
    """Check every archive's gene order, count provenance, shape, and sample rows."""
    genes_bytes = Path(genes_path).read_bytes()
    with Path(genes_path).open(newline="") as handle:
        genes = [row["gene"] for row in csv.DictReader(handle)]
    vocabulary_sha = hashlib.sha256(genes_bytes).hexdigest()
    if len(genes) != dataset.expected_genes or len(set(genes)) != len(genes):
        raise ValueError("gene vocabulary is not the expected unique 5,000-gene list")
    with Path(manifest).open(newline="") as handle:
        records = [r for r in csv.DictReader(handle) if r["split"] == split]
    if len(records) != len(dataset.archives):
        raise ValueError("manifest/archive count mismatch")
    import cellstream

    audited = []
    for i, (path, row) in enumerate(tqdm(list(zip(dataset.archives, records)), desc=f"Audit {split}", unit="archive")):
        store = cellstream.open(path)
        try:
            expected = int(float(row["selected_cell_count"]))
            if tuple(store.shape) != (expected, len(genes)):
                raise ValueError(f"shape mismatch: {path}: {store.shape}")
            if store.var.index.astype(str).tolist() != genes:
                raise ValueError(f"gene order mismatch: {path}")
            provenance = store.uns.get("celltok_stage1", {})
            if provenance.get("gene_vocabulary_sha256") != vocabulary_sha:
                raise ValueError(f"gene vocabulary digest mismatch: {path}")
            selection = provenance.get("cell_selection")
            if selection is not None and selection != "primary":
                raise ValueError(f"not a primary archive: {path}")
            # Earlier converter versions predate cell_selection metadata. Check
            # the actual stored observations rather than trusting that flag.
            if "is_primary_data" in store.obs:
                if not (store.obs["is_primary_data"] == True).all():
                    raise ValueError(f"archive contains non-primary observations: {path}")
                primary_evidence = "all obs.is_primary_data are True"
            elif selection == "primary" or provenance.get("primary_only") is True:
                primary_evidence = "converter primary selection provenance"
            elif row.get("has_is_primary_data") == "False":
                primary_evidence = "source has no primary flag; manifest retains all rows"
            else:
                raise ValueError(f"cannot establish primary selection for {path}: {provenance}")
            if provenance.get("storage") != "counts":
                raise ValueError(f"raw integer counts required: {path}")
            local = np.unique(np.linspace(0, expected-1, min(3, expected), dtype=np.int64)).tolist()
            sparse = store.gather_rows(local, n_threads=1)
            values = sparse.data
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.rint(values)):
                raise ValueError(f"invalid raw counts: {path}")
            raw = sparse.toarray().astype(np.float32)
            sums = raw.sum(axis=1, keepdims=True)
            normalized = np.log1p(raw * np.divide(10000.0, sums, out=np.zeros_like(sums), where=sums>0))
            indices = [dataset.offsets[i] + j for j in local]
            actual = dataset.get_batch_numpy(indices)
            np.testing.assert_allclose(actual, normalized, atol=2e-6, rtol=2e-6)
            audited.append({"dataset_id": row["dataset_id"], "cells": expected,
                            "archive": row["cellstream_path"], "primary_evidence": primary_evidence})
        finally:
            store.close()
            dataset.close()
    return {"split": split, "cells": len(dataset), "archives": audited,
            "gene_vocabulary_sha256": vocabulary_sha, "preprocessing": "log1p(10000 * counts / row_sum_of_aligned_5000_genes)"}
