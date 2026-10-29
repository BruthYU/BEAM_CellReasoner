"""CPU regression tests for AdamW scheduling, checkpoint cadence and resume."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from cellstream.cell import CellArchiveWriter
from torch.utils.data import DataLoader

from lib_vqvae.advanced_training import warmup_cosine_factor
from lib_vqvae.cellstream_dataset import CellStreamDataset
from lib_vqvae.vqvae import VQVAE
from celltok.data.training import audit_dataset


class AdvancedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)
        self.temporary = tempfile.TemporaryDirectory(prefix="celltok-advanced-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def model():
        return VQVAE(5, n=2, codebook_dim=2, n_codebooks=4,
                     encoder_hidden_dim=[6], decoder_hidden_dim=[6], dropout_rate=0)

    def test_warmup_and_cosine_endpoints(self):
        factors = [warmup_cosine_factor(s, 10, 2, 0.1) for s in range(10)]
        self.assertEqual(factors[:2], [0.5, 1.0])
        self.assertAlmostEqual(factors[-1], 0.1)
        self.assertTrue(all(a >= b for a,b in zip(factors[1:], factors[2:])))
        self.assertEqual(warmup_cosine_factor(0, 10, 0), 1.0)
        self.assertAlmostEqual(warmup_cosine_factor(9, 10, 0), 0.1)

    def test_adamw_checkpoint_every_five_and_exact_epoch_resume(self):
        model = self.model()
        initial = model.encoder.network[0].weight.detach().clone()
        loader = DataLoader(torch.rand(512, 5), batch_size=512, drop_last=True)
        args = SimpleNamespace(outdir=str(self.root/"full"))
        results = model.fit_ddp_advanced(args, loader, None, loader, "vqvae", "cpu", max_iter=6)
        self.assertFalse(torch.equal(initial, model.encoder.network[0].weight))
        self.assertFalse((self.root/"full/epoch_1").exists())
        self.assertTrue((self.root/"full/epoch_5/model.pth").exists())
        self.assertTrue((self.root/"full/epoch_6/model.pth").exists())
        checkpoint = torch.load(self.root/"full/epoch_6/model.pth", weights_only=False)
        self.assertEqual(checkpoint["advanced_settings"]["optimizer"], "AdamW")
        self.assertEqual(checkpoint["global_step"], 6)
        self.assertEqual(checkpoint["scheduler"]["last_epoch"], 6)
        self.assertEqual(len(results["eval_loss"]), 6)
        self.assertTrue(all(int(s["step"]) == 6 for s in checkpoint["optimizer"]["state"].values()))
        resumed = self.model()
        resume_args = SimpleNamespace(outdir=str(self.root/"resumed"))
        resumed.fit_ddp_advanced(resume_args, loader, None, loader, "vqvae", "cpu", max_iter=6,
                                 resume=self.root/"full/epoch_5/model.pth")
        for actual, expected in zip(resumed.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        progress = json.loads((self.root/"resumed/progress.json").read_text())
        self.assertEqual(progress["completed_steps"], 6)
        self.assertEqual(progress["status"], "complete")
        self.assertEqual(progress["eta_seconds"], 0)

    def test_sgd_plateau_epoch_steps_and_exact_resume(self):
        model = self.model()
        loader = DataLoader(torch.rand(1024, 5), batch_size=512, drop_last=True)
        options = dict(max_iter=3, lr=1e-4, optimizer_name="sgd",
                       scheduler_name="reduce_on_plateau", momentum=.9,
                       plateau_factor=.5, plateau_patience=0, log_val=1,
                       eval_every_steps=1, warmup_ratio=0)
        args = SimpleNamespace(outdir=str(self.root/"sgd"))
        with patch("lib_vqvae.advanced_training.advanced_validation", return_value=1.):
            result = model.fit_ddp_advanced(args, loader, None, loader, "vqvae", "cpu", **options)
        checkpoint = torch.load(self.root/"sgd/epoch_3/model.pth", weights_only=False)
        self.assertEqual(checkpoint["advanced_settings"]["optimizer"], "SGD")
        self.assertEqual(checkpoint["global_step"], 6)
        self.assertEqual(checkpoint["scheduler"]["last_epoch"], 3)
        self.assertEqual([r["learning_rate"] for r in result["history"]], [1e-4, 1e-4, 5e-5])
        self.assertEqual(result["history"][-1]["next_learning_rate"], 2.5e-5)
        self.assertTrue(all("momentum_buffer" in s for s in checkpoint["optimizer"]["state"].values()))
        resumed = self.model()
        with patch("lib_vqvae.advanced_training.advanced_validation", return_value=1.):
            resumed.fit_ddp_advanced(SimpleNamespace(outdir=str(self.root/"sgd-resume")),
                loader, None, loader, "vqvae", "cpu",
                resume=self.root/"sgd/epoch_1/model.pth", **options)
        for actual, expected in zip(resumed.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            self.model().fit_ddp_advanced(SimpleNamespace(outdir=str(self.root/"wrong-optimizer")),
                loader, None, loader, "vqvae", "cpu", max_iter=3, lr=1e-4,
                warmup_ratio=0, scheduler_name="reduce_on_plateau",
                resume=self.root/"sgd/epoch_1/model.pth")


def test_training_cli_prepares_data_and_calls_model_fit(tmp_path):
    """Exercise the maintained frontend on tiny CPU archives, not the live run."""
    import subprocess
    import sys
    import yaml
    genes = tmp_path/"genes.csv"
    genes.write_text("gene\ng0\ng1\ng2\n")
    rows = []
    for split in ("train", "test"):
        archive = tmp_path/f"{split}.csad"
        counts = sp.csr_matrix(np.tile([[1, 3, 2]], (1024, 1)), dtype=np.uint32)
        with CellArchiveWriter(archive, n_vars=3, value_dtype=np.uint32) as writer:
            writer.add_block(counts)
            writer.finalize(obs=pd.DataFrame({"is_primary_data": [True]*1024}, index=[f"{split}-{i}" for i in range(1024)]),
                            var=pd.DataFrame(index=["g0", "g1", "g2"]),
                            uns={"celltok_stage1": {"storage": "counts", "cell_selection": "primary",
                                 "gene_vocabulary_sha256": hashlib.sha256(genes.read_bytes()).hexdigest()}})
        rows.append(dict(dataset_id=split, study_id=split, split=split,
                         cellstream_path=str(archive), selected_cell_count="1024"))
    manifest = tmp_path/"manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    config = {"seed": 24, "data": {"train_manifest": str(manifest), "validation_manifest": str(manifest),
              "train_split": "train", "validation_split": "test", "genes": str(genes), "expected_genes": 3,
              "target_sum": 10000., "log_transform": "log1p", "local_cache_dir": str(tmp_path/"cache"),
              "num_workers_per_rank": 0},
              "model": {"data_dim": 3, "n": 2, "codebook_dim": 2, "n_codebooks": 4,
                        "encoder_hidden_dim": [4], "decoder_hidden_dim": [4], "beta": 1., "gamma": 1.,
                        "dropout_rate": 0., "mode": "vqvae"},
              "training": {"epochs": 2, "global_batch_size": 512, "save_every_epochs": 1,
                           "output_dir": str(tmp_path/"run")}}
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    completed = subprocess.run([sys.executable, "-m", "celltok", "train", "--config", str(path),
                                "--device", "cpu", "--num-gpus", "1", "--learning-rate", "0.0003", "--max-train-steps", "1"],
                               text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    result = json.loads((tmp_path/"run/result.json").read_text())
    assert result["weights_updated"] and result["checkpoint_reloaded"]
    assert result["global_step"] == 2 and result["optimizer"] == "AdamW"


class FitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="celltok-fit-test-")
        self.root = Path(self.temporary.name)
        torch.set_num_threads(1)
        torch.manual_seed(4)

    def tearDown(self):
        self.temporary.cleanup()

    def check_fit(self, steps):
        model = VQVAE(5, n=2, codebook_dim=2, n_codebooks=4,
                      encoder_hidden_dim=[6], decoder_hidden_dim=[6], dropout_rate=0.2)
        model.eval()  # fit must restore training mode on entry.
        before = model.encoder.network[0].weight.detach().clone()
        loader = DataLoader(torch.rand(steps*4, 5), batch_size=4)
        args = SimpleNamespace(outdir=str(self.root), log_val=10)
        results = model.fit(args, loader, None, loader, "vqvae", torch.device("cpu"), max_iter=1)
        self.assertFalse(torch.equal(before, model.encoder.network[0].weight))
        self.assertTrue(model.training)
        self.assertEqual(len(results["eval_loss"]), 1)
        checkpoint = torch.load(self.root/"epoch_1/model.pth", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], 3e-4)
        self.assertEqual(checkpoint["results"]["n_updates"], [1])
        progress = json.loads((self.root/"progress.json").read_text())
        self.assertEqual(progress["status"], "complete")
        self.assertEqual(progress["completed_steps"], steps)
        self.assertEqual(progress["eta_seconds"], 0)
        metrics = json.loads((self.root/"metrics.jsonl").read_text())
        self.assertEqual(metrics["cells"], steps*4)
        self.assertGreater(metrics["total_loss"], 0)

    def test_short_fit_validates_updates_and_saves(self):
        self.check_fit(2)

    def test_step_100_validation_and_checkpoint_reload(self):
        self.check_fit(101)

    def test_save_false(self):
        model = VQVAE(5, n=2, codebook_dim=2, n_codebooks=4,
                      encoder_hidden_dim=[6], decoder_hidden_dim=[6])
        loader = DataLoader(torch.rand(8, 5), batch_size=4)
        args = SimpleNamespace(outdir=str(self.root), log_val=1)
        model.fit(args, loader, None, loader, "vqvae", "cpu", max_iter=1, save=False)
        self.assertFalse(list(self.root.glob("epoch_*")))
        self.assertEqual(json.loads((self.root/"progress.json").read_text())["status"], "complete")

    def test_cellstream_counts_normalization_and_bounded_handles(self):
        paths = [self.root/"first.csad", self.root/"second.csad"]
        counts = sp.csr_matrix([[1, 3, 0], [0, 0, 0]], dtype=np.uint32)
        for path in paths:
            with CellArchiveWriter(path, n_vars=3, value_dtype=np.uint32) as writer:
                writer.add_block(counts)
                writer.finalize(obs=pd.DataFrame(index=["cell0", "cell1"]),
                                var=pd.DataFrame(index=["g0", "g1", "g2"]))
        dataset = CellStreamDataset(paths, expected_genes=3, max_open_archives=1)
        try:
            actual = dataset.get_batch_numpy([0, 1, 2, 3])
            expected = np.log1p([[2500, 7500, 0], [0, 0, 0]]).astype(np.float32)
            np.testing.assert_allclose(actual, np.concatenate([expected, expected]), rtol=1e-6)
            self.assertEqual(len(dataset._stores), 1)
            np.testing.assert_allclose(dataset.get_batch_numpy([0, 1]), expected, rtol=1e-6)
            self.assertEqual(len(dataset._stores), 1)
        finally:
            dataset.close()

    def test_audit_legacy_primary_metadata_checks_actual_observations(self):
        genes = self.root/"genes.csv"
        genes.write_text("gene\ng0\ng1\ng2\n")
        path = self.root/"legacy.csad"
        counts = sp.csr_matrix([[1, 3, 0], [0, 0, 0]], dtype=np.uint32)
        with CellArchiveWriter(path, n_vars=3, value_dtype=np.uint32) as writer:
            writer.add_block(counts)
            writer.finalize(obs=pd.DataFrame({"is_primary_data": [True, True]}, index=["c0", "c1"]),
                            var=pd.DataFrame(index=["g0", "g1", "g2"]),
                            uns={"celltok_stage1": {"storage": "counts",
                                "gene_vocabulary_sha256": hashlib.sha256(genes.read_bytes()).hexdigest()}})
        manifest = self.root/"manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset_id", "split", "cellstream_path", "selected_cell_count"])
            writer.writeheader()
            writer.writerow({"dataset_id": "test", "split": "train", "cellstream_path": str(path), "selected_cell_count": 2})
        dataset = CellStreamDataset([path], expected_genes=3)
        try:
            audit = audit_dataset(dataset, manifest, "train", genes)
            self.assertEqual(audit["cells"], 2)
            self.assertEqual(audit["archives"][0]["primary_evidence"], "all obs.is_primary_data are True")
        finally:
            dataset.close()
