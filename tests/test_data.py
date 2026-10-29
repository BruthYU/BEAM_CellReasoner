"""Sparse conversion, complete manifests, and reproducible batched loading."""
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from cellstream.cell import CellArchiveWriter

from lib_vqvae.cellstream_dataset import (
    CellStreamDataset,
    normalize_expression_batch,
    read_archive_manifest,
    stage_archives_locally,
)
from lib_vqvae.metrics import codebook_usage_metrics
from lib_vqvae.vqvae import VQVAE
from lib_vqvae.samplers import ArchiveBatchSampler
from celltok.data.convert import (
    align_expression_block,
    main as convert_h5ad_main,
    sanitize_obs_columns,
    select_matrix,
)
from celltok.data.freeze import freeze_manifest


def test_frozen_manifest_paths_work_after_legacy_directories_are_removed(tmp_path):
    from lib_vqvae.paths import resolve_data_path

    data = tmp_path / "data"
    canonical = data / "stage1/full_primary"
    canonical.mkdir(parents=True)
    archive = canonical / "tiny.csad"
    write_test_archive(archive)
    manifest = canonical / "frozen.csv"
    old_root = data / "celltok_stage1_shared"
    manifest.write_text(f"cellstream_path,split\n{old_root / archive.name},train\n")
    original_bytes = manifest.read_bytes()
    assert not old_root.exists()
    assert read_archive_manifest(old_root / manifest.name, split="train") == [archive]
    dataset = CellStreamDataset.from_manifest(old_root / manifest.name, split="train", expected_genes=3)
    try:
        assert dataset.get_batch_numpy([0, 3]).shape == (2, 3)
    finally:
        dataset.close()
    assert manifest.read_bytes() == original_bytes
    assert resolve_data_path("data/celltok_stage1/h5ad/a.h5ad", base_dir=tmp_path) == data / "stage1/legacy/h5ad/a.h5ad"
    assert resolve_data_path("data/celltok_stage1_shared_nonprimary/a.csad", base_dir=tmp_path) == data / "stage1/nonprimary/a.csad"
    unrelated = data / "celltok_stage1_other/a.csad"
    assert resolve_data_path(unrelated) == unrelated


def test_normalization_log_variants():
    counts = sp.csr_matrix([[1, 3, 0], [0, 0, 0]], dtype=np.uint32)
    log1p = normalize_expression_batch(counts, target_sum=100, log_transform="log1p")
    log10 = normalize_expression_batch(counts, target_sum=100, log_transform="log10")
    np.testing.assert_allclose(log1p[0], np.log1p([25, 75, 0]), rtol=1e-6)
    np.testing.assert_allclose(log10[0], np.log10(1 + np.array([25, 75, 0])), rtol=1e-6)
    np.testing.assert_array_equal(log1p[1], 0)


def write_test_archive(path: Path):
    counts = sp.csr_matrix(
        [[1, 0, 3], [0, 2, 0], [4, 5, 0], [1, 1, 1]], dtype=np.uint32
    )
    obs = pd.DataFrame(
        {"cell_type": ["a", "b", "a", "b"]},
        index=[f"cell-{index}" for index in range(4)],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    with CellArchiveWriter(path, n_vars=3, value_dtype=np.uint32) as writer:
        writer.add_block(counts[:2])
        writer.add_block(counts[2:])
        writer.finalize(obs=obs, var=var)


def test_cellstream_dataset_batched_gather(tmp_path):
    archive = tmp_path / "tiny.csad"
    write_test_archive(archive)
    dataset = CellStreamDataset(
        [archive], expected_genes=3, target_sum=10, log_transform="none", read_threads=2
    )
    batch = dataset.get_batch_numpy([3, 0, 2])
    np.testing.assert_allclose(batch.sum(axis=1), 10)
    assert dataset.get_obs_values([3, 0], "cell_type").tolist() == ["b", "a"]
    tensors = dataset.__getitems__([1, 2])
    assert len(tensors) == 2 and tensors[0].shape == (3,)
    dataset.close()


def test_archive_batch_sampler_is_local_reproducible_and_distributed():
    lengths = [5, 7, 4]
    rank0 = ArchiveBatchSampler(
        lengths, 2, seed=17, num_replicas=2, rank=0, drop_last=True
    )
    rank1 = ArchiveBatchSampler(
        lengths, 2, seed=17, num_replicas=2, rank=1, drop_last=True
    )
    batches0 = list(rank0)
    batches1 = list(rank1)
    assert len(batches0) == len(batches1) == 3
    assert not ({tuple(batch) for batch in batches0} & {tuple(batch) for batch in batches1})
    offsets = np.cumsum([0, *lengths])
    for batch in batches0 + batches1:
        archive_ids = {
            int(np.searchsorted(offsets, index, side="right") - 1)
            for index in batch
        }
        assert len(archive_ids) == 1
    assert batches0 == list(
        ArchiveBatchSampler(
            lengths, 2, seed=17, num_replicas=2, rank=0, drop_last=True
        )
    )
    rank0.set_epoch(1)
    assert batches0 != list(rank0)


def test_archive_handles_are_bounded_without_changing_rows(tmp_path):
    paths = [tmp_path / "first.csad", tmp_path / "second.csad"]
    for path in paths:
        write_test_archive(path)
    dataset = CellStreamDataset(paths, expected_genes=3, max_open_archives=1)
    first = dataset.get_batch_numpy([0, 1])
    second = dataset.get_batch_numpy([4, 5])
    assert len(dataset._stores) == 1
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(dataset.get_batch_numpy([0, 1]), first)
    assert len(dataset._stores) == 1
    dataset.close()


def test_evaluation_codebook_usage_summary():
    indices = np.asarray([[0, 1], [1, 1], [1, 2]], dtype=np.int64)
    summary = codebook_usage_metrics(indices, n_codes=4)
    assert summary["cells"] == 3
    assert summary["positions"] == 2
    assert summary["token_count"] == 6
    assert summary["active_codes"] == 3
    assert summary["code_utilization"] == 0.75
    assert summary["active_codes_by_position"] == [2, 2]
    assert summary["counts_by_code"] == [1, 4, 1, 0]
    np.testing.assert_allclose(summary["code_perplexity"], 2.381101577952299)


def test_local_staging_reuses_existing_archive(tmp_path, monkeypatch):
    source = tmp_path / "tiny.csad"
    write_test_archive(source)
    cache = tmp_path / "cache"
    staged = stage_archives_locally([source], cache)
    monkeypatch.setattr(
        "lib_vqvae.cellstream_dataset.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": 0})(),
    )
    assert stage_archives_locally([source], cache) == staged


def test_eval_does_not_update_quantizer_state():
    model = VQVAE(
        data_dim=3,
        n=2,
        codebook_dim=2,
        n_codebooks=4,
        encoder_hidden_dim=[4],
        decoder_hidden_dim=[4],
        beta=1,
        gamma=1,
        dropout_rate=0,
    )
    model.vector_quantization.anchor = "random"
    model.eval()
    before = model.vector_quantization.embedding.weight.detach().clone()
    model(torch.rand(3, 3), "cvqvae", torch.device("cpu"))
    torch.testing.assert_close(model.vector_quantization.embedding.weight, before)


def test_raw_or_x_matrix_fallback():
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 2]], dtype=np.float32),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    matrix, var, source = select_matrix(adata, "raw-or-X")
    assert matrix is adata.X and var is adata.var and source == "X"

    adata.raw = adata.copy()
    matrix, var, source = select_matrix(adata, "raw-or-X")
    assert source == "raw"
    np.testing.assert_array_equal(matrix.toarray(), [[1, 2]])


def test_obs_column_sanitization_is_safe_unique_and_reversible():
    frame = pd.DataFrame(
        [[1, 2, 3, 4]],
        columns=["Region/Tooth #", "Region__slash__Tooth #", "", ""],
    )
    safe, renames = sanitize_obs_columns(frame)
    assert safe.columns.tolist() == [
        "Region__slash__Tooth #",
        "Region__slash__Tooth #__2",
        "unnamed",
        "unnamed__2",
    ]
    assert len(set(safe.columns)) == len(safe.columns)
    assert all("/" not in value and value for value in safe.columns)
    assert [entry["position"] for entry in renames] == [0, 1, 2, 3]


def test_h5ad_converter_selected_cell_range(tmp_path, monkeypatch):
    source = tmp_path / "source.h5ad"
    output = tmp_path / "shard.csad"
    genes = tmp_path / "genes.csv"
    genes.write_text("gene\ng1\ng2\ng3\n")
    obs = pd.DataFrame(
        {"is_primary_data": [True, False, True, True, False, True]},
        index=[f"cell-{index}" for index in range(6)],
    )
    ad.AnnData(
        X=sp.csr_matrix(np.arange(18, dtype=np.uint32).reshape(6, 3)),
        obs=obs,
        var=pd.DataFrame(
            {"feature_name": ["g1", "g2", "g3"]},
            index=["gene-1", "gene-2", "gene-3"],
        ),
    ).write_h5ad(source)
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert_h5ad_to_cellstream.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--genes",
            str(genes),
            "--matrix",
            "X",
            "--storage",
            "counts",
            "--cell-selection",
            "primary",
            "--selected-cell-start",
            "1",
            "--selected-cell-stop",
            "3",
            "--block-rows",
            "1",
        ],
    )
    convert_h5ad_main()

    import cellstream

    store = cellstream.open(output)
    try:
        assert tuple(store.shape) == (2, 3)
        assert store.obs.index.tolist() == ["cell-2", "cell-3"]
        np.testing.assert_array_equal(
            store.gather_rows([0, 1]).toarray(),
            np.arange(18, dtype=np.uint32).reshape(6, 3)[[2, 3]],
        )
    finally:
        store.close()


def test_backed_csr_alignment_never_materializes_full_matrix(tmp_path, monkeypatch):
    from anndata._core.sparse_dataset import BaseCompressedSparseDataset

    counts = np.arange(48, dtype=np.uint32).reshape(8, 6)
    source = tmp_path / "backed.h5ad"
    ad.AnnData(X=sp.csr_matrix(counts)).write_h5ad(source)
    backed = ad.read_h5ad(source, backed="r")

    def forbidden_to_memory(_self):
        raise AssertionError("full-matrix materialization is forbidden")

    monkeypatch.setattr(BaseCompressedSparseDataset, "to_memory", forbidden_to_memory)
    try:
        actual = align_expression_block(
            backed.X,
            slice(2, 6),
            np.array([1, 4]),
            np.array([2, 0]),
            3,
            np.array([True, False, True, False]),
            np.dtype(np.uint32),
        )
        expected = np.zeros((2, 3), dtype=np.uint32)
        expected[:, 2] = counts[[2, 4], 1]
        expected[:, 0] = counts[[2, 4], 4]
        np.testing.assert_array_equal(actual.toarray(), expected)
    finally:
        backed.file.close()


class FullPrimaryTests(unittest.TestCase):
    def test_complete_shards_cover_all_cells_without_double_counting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ("test.csad","part0.csad","part1.csad"):
                (root/name).write_bytes(b"fixture")
            rows=[dict(dataset_id="a",split="train",study_id="study-a",selected_cell_count="20",cellstream_path="absent.csad"),
                  dict(dataset_id="b",split="test",study_id="study-b",selected_cell_count="10",cellstream_path="test.csad")]
            parts=[]
            for i in range(2):
                part=dict(rows[0],cellstream_path=f"part{i}.csad",selected_cell_count="10",selected_cell_start=str(i*10),
                          selected_cell_stop=str((i+1)*10),source_dataset_selected_cell_count="20",cell_selection="primary")
                parts.append(part)
            def write(path, values):
                with path.open("w",newline="") as handle:
                    w=csv.DictWriter(handle,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
            write(root/"manifest.csv",rows);write(root/"shards.csv",parts)
            with contextlib.redirect_stdout(io.StringIO()):
                freeze_manifest(root/"manifest.csv",root/"shards.csv",root/"full.csv",require_all=True)
            state=json.loads((root/"full.summary.json").read_text())
            self.assertTrue(state["all_target_cells_covered"])
            self.assertEqual(state["selected_cells"],{"train":20,"test":10})
            self.assertEqual(state["archives"],3)
            # A discontinuity must not silently become a partially covered set.
            parts[1]["selected_cell_start"]="11"
            parts[1]["selected_cell_count"]="9"
            write(root/"shards.csv",parts)
            with self.assertRaises(ValueError):
                freeze_manifest(root/"manifest.csv",root/"shards.csv",root/"bad.csv",require_all=True)
            self.assertFalse((root/"bad.csv").exists())
