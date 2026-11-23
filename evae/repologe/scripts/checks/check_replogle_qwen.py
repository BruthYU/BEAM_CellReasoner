"""CPU regression of Qwen presets, prompt equivalence, sharding and nine metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap

import gc
import json
import tempfile
import numpy as np
import h5py
from replogle_llm_common import ROOT, DEFAULT_ART, prepare_tokenizer, CellDataset, CellCollator, read_json, save_json
from replogle_qwen import PRESETS, preset_config, training_data, make_dataset, verify_inputs
from replogle_control_common import ControlCellDataset
from replogle_control_baseline import BaselineControlDataset
from replogle_control_c import ControlSetDataset
from eval_replogle_qwen import case_rows, eval_store, batch_positions, merge_shards, input_control_rows

ARCHIVE = ROOT / "artifacts/repository_cleanup/20261123_qwen_consolidation"


def main():
    tok = prepare_tokenizer(DEFAULT_ART / "initial_text_model")
    report = {"status": "passed", "presets": {}, "jobs_submitted": 0, "gpu_runtime_tested": False}
    for preset in PRESETS:
        cfg = preset_config(preset)
        verify_inputs(cfg)
        store, fold, sources, monitor = training_data(cfg)
        # Include non-targeting targets, every training gem and every held-out line.
        if preset == "original-text":
            ids = np.concatenate([np.flatnonzero(fold == 0)[:12], monitor[:12]])
            original = CellDataset(store, ids, tok)
        else:
            ids = np.unique(np.concatenate([np.flatnonzero(store.support & (fold == 0))[:12],
                [np.flatnonzero((store.gem == g) & (fold == 0))[0] for g in np.unique(store.gem[fold == 0])], monitor[:12]]))
            original = (BaselineControlDataset(store, ids, tok) if preset == "original-control1" else
                ControlSetDataset(store, ids, tok, cfg["control_cells"]) if preset == "hepg2-c" else
                ControlCellDataset(store, ids, tok, preset[-1].upper()))
        actual = make_dataset(cfg, store, ids, tok)
        assert original.max_length == actual.max_length <= cfg["max_sequence_length"]
        for i in range(len(ids)):
            assert original[i] == actual[i], (preset, i)
            assert sum(v != -100 for v in actual[i]["labels"]) == 66
            expected = original.inference_prefix(i) if hasattr(original, "inference_prefix") else original.prefixes[int(original.condition_rows[ids[i]])]
            assert actual.inference_prefix(i) == expected
        batch = CellCollator(tok)([actual[i] for i in range(min(8, len(ids)))])
        assert batch["labels"].shape[1] == 67
        if preset != "original-text":
            train_ids = ids[fold[ids] == 0]
            train = make_dataset(cfg, store, train_ids, tok, training=True)
            train.set_epoch(2)
            sampled = input_control_rows(train, train_ids, cfg["control_cells"])
            if cfg["control_cells"]:
                assert (fold[sampled] == 0).all()
                assert (sampled.reshape(len(train_ids), -1) != train_ids[:, None]).all()
        evaluation_store = eval_store(cfg)
        membership, union, cases = case_rows(cfg, evaluation_store)
        report["presets"][preset] = dict(train_cells=int((fold == 0).sum()), monitor_cells=len(monitor),
            checked_prompt_rows=len(ids), control_cells=cfg["control_cells"], evaluation_cells=len(union), cases=cases)
        del store, actual, original, evaluation_store
        gc.collect()
    # C remains configurable and the N=1 prompt retains C's template, not B's.
    for n in (1, 8, 32):
        assert preset_config("hepg2-c", control_cells=n)["control_cells"] == n
    for preset, n in (("original-text", 1), ("hepg2-b", 8), ("hepg2-c", 0)):
        try: preset_config(preset, control_cells=n)
        except ValueError: pass
        else: raise AssertionError("Incompatible preset/count accepted")
    with tempfile.TemporaryDirectory(dir=ARCHIVE, prefix="cpu-regression-") as tmp:
        root = Path(tmp)
        # N=0, N=1 and N=8 control provenance; 1/2/4/8 generation workers.
        for world in (1, 2, 4, 8):
            for n in (0, 1, 8):
                out = root / f"g{world}-n{n}"; (out / "shards").mkdir(parents=True)
                rows = np.arange(137, dtype=np.int64) * 3 + 7
                plan = dict(eval_gpus=world, batch_size=16, control_cells=n, model_sha256="synthetic")
                with h5py.File(out / "real_cells.h5", "w") as h: h["global_rows"] = rows
                for rank in range(world):
                    pos = np.concatenate([p for _, p in batch_positions(len(rows), 16, rank, world)])
                    with h5py.File(out / "shards" / f"rank-{rank:02d}.h5", "w") as h:
                        h.attrs.update(complete=True, rank=rank, world_size=world, model_sha256="synthetic")
                        h["positions"] = pos; h["global_rows"] = rows[pos]
                        h["control_input_row"] = pos + 1000 if n == 1 else np.repeat(pos[:, None] + 1000, n, axis=1)
                        h["predicted_counts"] = np.repeat(pos[:, None], 2000, axis=1).astype(np.float32)
                        h["generated_codes"] = np.repeat(pos[:, None] % 512, 64, axis=1).astype(np.uint16)
                merge_shards(out, plan)
                with h5py.File(out / "generated_cells.h5", "r") as h:
                    np.testing.assert_array_equal(h["predicted_counts"][:, 0], np.arange(len(rows)))
                    assert h["control_input_row"].shape == ((len(rows),) if n == 1 else (len(rows), n))
                (out / "generation_complete.json").unlink()
                with h5py.File(out / "shards/rank-00.h5", "r+") as h: h["positions"][0] = 1
                try: merge_shards(out, plan)
                except AssertionError: pass
                else: raise AssertionError("Corrupted shard accepted")
        # Exercise the real metric bridge using a synthetic identity prediction.
        import torch, yaml
        from replogle_qwen_metrics import evaluate_case
        from replogle_control_eval_common import nine_metrics
        from replogle_control_baseline import ART
        torch.set_num_threads(4)
        source = read_json(ART / "sources.json")
        config = yaml.safe_load((ROOT / source["vae_run"] / "config.yaml").read_text())
        config["_vae_run"] = source["vae_run"]
        labels = read_json(ROOT / source["nb_cache"] / "labels.json")
        genes = (ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt").read_text().splitlines()
        symbols = (ROOT / "assets/replogle_scldm2k_gene_order.symbols.txt").read_text().splitlines()
        selected = sorted(set(symbols) & set(labels["perturbations"]))[:2]
        counts = np.random.default_rng(420).poisson(3, (192, 2000)).astype(np.float32)
        counts[64:128, :100] += 4; counts[128:, 100:200] += 6
        pert = np.repeat([labels["perturbations"].index(g) for g in ["non-targeting"] + selected], 64)
        ct = np.full(192, labels["cell_types"].index("hepg2"))
        case = dict(key="identity", split="fixture", cell_line="HepG2", target_cells=128, conditions=2, reference_cells=64)
        metrics = nine_metrics(evaluate_case(case, root, counts, counts.copy(), pert, ct, labels, genes, np.arange(192), config, smoke=True))
        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics["W2"] < 1e-4 and abs(metrics["MMD2"]) < 1e-5 and abs(metrics["FD"]) < 1e-4
        assert metrics["PDS"] == 1 and metrics["P-delta"] > .9999
        report["synthetic_identity_nine_metrics"] = metrics
    report["sharding"] = "12 combinations of 1/2/4/8 workers and 0/1/8 controls; corrupt shards rejected"
    save_json(ARCHIVE / "cpu_regression.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
