"""Metric identities, evaluation comparability, and observed epoch reports."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from lib_vqvae.metrics import clustering_metrics, rbf_mmd, reconstruction_metrics
from celltok.report import assert_same_protocol, compare, load_history, official_history, parse_named
from celltok.evaluation.reconstruction import StreamingReconstructionMetrics


@pytest.mark.parametrize("block_size", [1, 4, 11])
def test_streaming_metrics_match_reference_with_uneven_batches(block_size):
    rng = np.random.default_rng(19)
    original = rng.normal(size=(17, 7)).astype(np.float32)
    reconstructed = rng.normal(size=(17, 7)).astype(np.float32)
    original[0] = 0
    reconstructed[0] = 2
    original[1] = 7
    reconstructed[1] = 3
    original[:, -1] = reconstructed[:, -1] = 0
    accumulator = StreamingReconstructionMetrics(7)
    for start in range(0, 17, block_size):
        accumulator.update(original[start:start+block_size], reconstructed[start:start+block_size])
    expected = reconstruction_metrics(original, reconstructed)
    for key, value in accumulator.result().items():
        assert value == pytest.approx(expected[key], rel=1e-12, abs=1e-12)


def test_streaming_metrics_zero_rows_and_input_validation():
    accumulator = StreamingReconstructionMetrics(3)
    with pytest.raises(ValueError, match="no cells"):
        accumulator.result()
    accumulator.update(np.zeros((2, 3)), np.zeros((2, 3)))
    assert all(value == 0 for value in accumulator.result().values())
    constant = StreamingReconstructionMetrics(3)
    constant.update(np.full((2, 3), 7.0), np.full((2, 3), 3.0))
    assert constant.result()["pcc"] == 0
    assert constant.result()["acs_cellwise"] == pytest.approx(1)
    with pytest.raises(ValueError, match="matching"):
        accumulator.update(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        accumulator.update(np.full((1, 3), np.nan), np.ones((1, 3)))


def test_full_reconstruction_cli_covers_all_archives_and_tail(tmp_path):
    import csv
    import hashlib
    import subprocess
    import sys
    import pandas as pd
    import scipy.sparse as sp
    import torch
    import yaml
    from cellstream.cell import CellArchiveWriter
    from lib_vqvae.cellstream_dataset import normalize_expression_batch
    from lib_vqvae.factory import build_vqvae

    genes = tmp_path/"genes.csv"
    genes.write_text("gene\ng0\ng1\ng2\n")
    rows, blocks = [], []
    for index, count in enumerate((7, 6)):
        archive = tmp_path/f"part{index}.csad"
        values = np.random.default_rng(index).integers(0, 20, size=(count, 3), dtype=np.uint32)
        blocks.append(values)
        with CellArchiveWriter(archive, n_vars=3, value_dtype=np.uint32) as writer:
            writer.add_block(sp.csr_matrix(values))
            writer.finalize(obs=pd.DataFrame({"is_primary_data": [True]*count}, index=[f"{index}-{i}" for i in range(count)]),
                            var=pd.DataFrame(index=["g0", "g1", "g2"]),
                            uns={"celltok_stage1": {"storage": "counts", "cell_selection": "primary",
                                 "gene_vocabulary_sha256": hashlib.sha256(genes.read_bytes()).hexdigest()}})
        rows.append(dict(dataset_id=str(index), study_id=str(index), split="test", selected_cell_count=count, cellstream_path=str(archive)))
    manifest = tmp_path/"manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    config = {"seed": 2024, "data": {"validation_manifest": str(manifest), "validation_split": "test",
              "genes": str(genes), "expected_genes": 3, "target_sum": 10000, "log_transform": "log1p",
              "local_cache_dir": str(tmp_path/"cache")},
              "model": {"data_dim": 3, "n": 2, "codebook_dim": 2, "n_codebooks": 4,
                        "encoder_hidden_dim": [4], "decoder_hidden_dim": [4], "dropout_rate": 0,
                        "beta": 10, "gamma": 5, "mode": "vqvae"},
              "evaluation": {"batch_size": 5, "max_cells": 7, "random_seed": 2024}}
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    torch.manual_seed(27)
    model = build_vqvae(config)
    checkpoint = tmp_path/"model.pth"
    torch.save({"model": model.state_dict()}, checkpoint)
    output = tmp_path/"metrics.json"
    run = subprocess.run([sys.executable, "-m", "celltok", "evaluate", "tokenizer", "--config", str(path),
                          "--checkpoint", str(checkpoint), "--device", "cpu", "--reconstruction-only",
                          "--all-cells", "--output", str(output)], capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout+run.stderr
    result = json.loads(output.read_text())
    assert result["cells"] == 13
    assert result["protocol"]["indices_sha256"] == hashlib.sha256(np.arange(13, dtype=np.int64).tobytes()).hexdigest()
    assert len(result["data_audit"]["archives"]) == 2
    original = normalize_expression_batch(sp.csr_matrix(np.concatenate(blocks)))
    model.eval()
    with torch.inference_mode():
        reconstructed = model(torch.from_numpy(original), "vqvae", torch.device("cpu"))[3].numpy()
    expected = reconstruction_metrics(original, reconstructed)
    for key, value in result["metrics"].items():
        assert value == pytest.approx(expected[key], rel=2e-6, abs=2e-7)
    progress = json.loads(output.with_suffix(".progress.json").read_text())
    assert progress["status"] == "complete" and progress["cells_done"] == 13


def test_reconstruction_metrics_identity():
    matrix = np.array([[1, 2, 3], [3, 1, 0]], dtype=np.float32)
    metrics = reconstruction_metrics(matrix, matrix.copy())
    assert metrics["mse"] == 0.0
    np.testing.assert_allclose(metrics["pcc"], 1.0)
    np.testing.assert_allclose(metrics["acs"], 1.0)
    np.testing.assert_allclose(metrics["acs_cellwise"], 1.0)
    np.testing.assert_allclose(metrics["acs_genewise"], 1.0)
    np.testing.assert_allclose(metrics["acs_global"], 1.0)
    mmd, _ = rbf_mmd(matrix, matrix.copy(), max_cells=2, bandwidth_sample=4)
    assert abs(mmd) < 1e-8


def test_clustering_metrics_separable_groups():
    representation = np.array(
        [[0, 0], [0, 0.1], [10, 10], [10.1, 10]], dtype=np.float32
    )
    metrics = clustering_metrics(representation, ["a", "a", "b", "b"], seed=1)
    assert metrics["ari"] == 1.0
    assert metrics["nmi"] == 1.0
    assert metrics["ami"] == 1.0


def test_protocol_requires_same_cells_and_indices():
    result = {"manifest": "same.csv", "split": "test", "cells": 10000,
              "normalization": "log1p", "protocol": {"indices_sha256": "same", "seed": 2024}}
    assert assert_same_protocol([result, copy.deepcopy(result)])["cells"] == 10000
    for field, value in (("cells", 9999), ("normalization", "log10")):
        other = copy.deepcopy(result)
        other[field] = value
        with pytest.raises(ValueError, match="protocol"):
            assert_same_protocol([result, other])
    other = copy.deepcopy(result)
    other["protocol"]["indices_sha256"] = "different"
    with pytest.raises(ValueError, match="protocol"):
        assert_same_protocol([result, other])


def test_named_inputs_reject_duplicate_labels():
    assert parse_named(["official=model.json"])["official"] == Path("model.json")
    with pytest.raises(ValueError):
        parse_named(["same=a.json", "same=b.json"])


def test_official_history_keeps_only_observed_nineteen_epochs(tmp_path):
    values = {key: [float(i) for i in range(1, 20)] for key in ("recon_loss", "embedding_loss", "train_loss")}
    values["n_updates"] = list(range(1, 20))
    path = tmp_path/"official.json"
    path.write_text(json.dumps({"results": values, "hyperparameters": {"max_iter": 20}}))
    rows = official_history(path)
    assert len(rows) == 19 and rows[-1]["epoch"] == 19


def test_live_or_resumed_epoch_history_and_gap_detection(tmp_path):
    path = tmp_path/"metrics.jsonl"
    records = [{"epoch": i, "reconstruction_loss": .2, "total_loss": .3} for i in (6, 7)]
    path.write_text("\n".join(json.dumps(r) for r in records)+"\n"+'{"epoch":')
    assert [r["epoch"] for r in load_history(tmp_path)] == [6, 7]
    records[-1]["epoch"] = 8
    path.write_text("\n".join(json.dumps(r) for r in records)+"\n")
    with pytest.raises(ValueError, match="epoch history"):
        load_history(tmp_path)


def test_generic_comparison_writes_baseline_deltas(tmp_path, monkeypatch):
    import celltok.report as reports
    import matplotlib.pyplot as plt
    monkeypatch.setattr(reports, "save_figure", lambda fig, path: plt.close(fig))
    inputs = {}
    for label, mse in (("official", .24), ("trained", .20)):
        path = tmp_path/f"{label}.json"
        path.write_text(json.dumps({"manifest": "same.csv", "split": "test", "cells": 100,
                                   "normalization": "log1p", "protocol": {"seed": 2024},
                                   "metrics": {"mse": mse, "pcc": .6}}))
        inputs[label] = path
    result = compare(inputs, tmp_path)
    assert result["deltas_versus_baseline"][0]["mse"] == pytest.approx(-.04)
    assert (tmp_path/"comparison.csv").exists()
