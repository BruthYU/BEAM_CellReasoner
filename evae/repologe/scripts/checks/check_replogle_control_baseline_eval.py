"""CPU acceptance of evaluation coverage, shard merging and the actual nine metrics."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch
import yaml

from replogle_llm_common import ROOT, read_json, save_json
from replogle_control_common import CellStore
from replogle_control_baseline import ART
from replogle_control_eval_common import nine_metrics
from eval_replogle_qwen import evaluation_rows, batch_positions, merge_shards


def run_checks():
    torch.set_num_threads(4)
    store = CellStore(ART)
    targets, controls, rows, cases = evaluation_rows(store)
    # Real target rows plus the original reference pool: no half-HepG2 control split.
    assert len(controls) == 4976 and len(rows) == 47244
    for world in [1, 2, 4, 8]:
        chunks = [pos for rank in range(world) for _, pos in batch_positions(len(rows), 64, rank, world)]
        np.testing.assert_array_equal(np.sort(np.concatenate(chunks)), np.arange(len(rows)))
        by_batch = {bid: pos.tolist() for rank in range(world)
                    for bid, pos in batch_positions(len(rows), 64, rank, world)}
        if world == 1:
            expected_batches = by_batch
        else:
            assert by_batch == expected_batches
    # Exercise the real HDF5 merge with out-of-order worker assignment and a final short batch.
    fixture_root = Path(tempfile.mkdtemp(prefix="eval_cpu_", dir=ART))
    merged = []
    for world in [1, 8]:
        out = fixture_root / f"merge-{world}"
        (out / "shards").mkdir(parents=True)
        n = 137
        global_rows = np.arange(n, dtype=np.int64) * 3 + 7
        with h5py.File(out / "real_cells.h5", "w") as h:
            h.create_dataset("global_rows", data=global_rows)
        plan = dict(eval_gpus=world, batch_size=16, model_sha256="CPU_SYNTHETIC_NOT_A_MODEL")
        for rank in range(world):
            positions = np.concatenate([pos for _, pos in batch_positions(n, 16, rank, world)])
            with h5py.File(out / "shards" / f"rank-{rank:02d}.h5", "w") as h:
                h.attrs.update(complete=True, rank=rank, world_size=world, model_sha256=plan["model_sha256"])
                h.create_dataset("positions", data=positions)
                h.create_dataset("global_rows", data=global_rows[positions])
                h.create_dataset("control_input_row", data=positions + 1234)
                h.create_dataset("predicted_counts", data=np.repeat(positions[:, None], 2000, axis=1).astype(np.float32))
                h.create_dataset("generated_codes", data=np.repeat(positions[:, None] % 512, 64, axis=1).astype(np.uint16))
        merge_shards(out, plan)
        with h5py.File(out / "generated_cells.h5", "r") as h:
            assert h.attrs["complete"]
            np.testing.assert_array_equal(h["global_rows"][:], global_rows)
            np.testing.assert_array_equal(h["control_input_row"][:], np.arange(n) + 1234)
            merged.append(h["predicted_counts"][:])
        merge_shards(out, plan)  # Completed merge is safely reusable.
    np.testing.assert_array_equal(*merged)
    # The merge must reject duplicate/wrong row coverage instead of averaging rank scores.
    bad = fixture_root / "merge-8"
    (bad / "generation_complete.json").unlink()
    with h5py.File(bad / "shards/rank-01.h5", "r+") as h:
        h["positions"][0] = 0
    try:
        merge_shards(bad, plan)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted shard accepted")
    from replogle_qwen_metrics import evaluate_case
    import sc_evae.metrics.distribution as distribution
    assert Path(distribution.__file__).resolve() == (ROOT / "src/sc-evae/metrics/distribution.py").resolve()
    source = read_json(ART / "sources.json")
    config = yaml.safe_load((ROOT / source["vae_run"] / "config.yaml").read_text())
    config["_vae_run"] = source["vae_run"]
    labels = read_json(ROOT / source["nb_cache"] / "labels.json")
    genes = (ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt").read_text().splitlines()
    symbols = (ROOT / "assets/replogle_scldm2k_gene_order.symbols.txt").read_text().splitlines()
    selected_genes = sorted(set(symbols) & set(labels["perturbations"]))[:2]
    counts = np.random.default_rng(420).poisson(3, (192, 2000)).astype(np.float32)
    counts[64:128, :100] += 4
    counts[128:, 100:200] += 6
    pert = np.repeat([labels["perturbations"].index(g) for g in ["non-targeting"] + selected_genes], 64)
    ct = np.full(192, labels["cell_types"].index("hepg2"))
    case = dict(key="synthetic_identity", split="cpu_fixture", cell_line="HepG2", target_cells=128,
                conditions=2, reference_cells=64)
    result = evaluate_case(case, fixture_root, counts, counts.copy(), pert, ct, labels, genes,
                           np.arange(192), config, smoke=True)
    values = nine_metrics(result)
    assert all(np.isfinite(v) for v in values.values())
    assert values["W2"] < 1e-4 and abs(values["MMD2"]) < 1e-5 and abs(values["FD"]) < 1e-4
    assert values["PDS"] == 1 and values["P-delta"] > .9999
    report = dict(status="passed", device="cpu", cases=cases, unique_generation_cells=len(rows),
        sharding="1/2/4/8 workers cover each global row exactly once; global batches unchanged",
        merge="1/8 worker synthetic merge identical; corrupted shard rejected",
        synthetic_identity_nine_metrics=values, gpu_generation="pending; no job submitted")
    save_json(ART / "eval_cpu_preflight.json", report)
    return report


if __name__ == "__main__":
    print(run_checks(), flush=True)
