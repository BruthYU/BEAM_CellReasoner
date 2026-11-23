"""Sharded generation, merged populations, unchanged nine metrics on the original split."""
# 常用：A/B/C 独立提交；将 --dry-run 换成 --submit 即提交。
# bash scripts/launch/submit_replogle_control_eval.sh --variant A --dry-run
# 常用：在已有GPU任务内评测/恢复新版训练结果。
# bash scripts/workers/run_replogle_qwen_eval.sh RUN_DIR
# 原文本模型直接评测（需GPU和项目环境）：
# python scripts/evaluation/eval_replogle_qwen.py --protocol legacy-text --model-dir MODEL --output-dir OUTPUT --split test
# 详细参数：--help；历史协议参数：--protocol legacy-control --help 或 --protocol legacy-text --help。


# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import argparse
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import yaml

from replogle_llm_common import ROOT, read_json, save_json, sha, install_fast_delta
from replogle_control_common import CellStore
from replogle_control_baseline import ART, safe_output as baseline_safe_output
from replogle_qwen import (make_dataset, verify_inputs as verify_preset, safe_output as preset_safe_output,
    AB_ART, DEFAULT_ART)
from replogle_control_eval_common import METRICS, nine_metrics
from replogle_qwen_inference import load_llm, load_nb_decoder, predict_cells


def original_evaluation_rows(store):
    lines = np.array([c["cell_line"] for c in store.conditions])[store.condition_rows]
    controls = np.flatnonzero(store.support & (lines == "HepG2"))
    # Original evaluator's 5000 cap does not subsample the 4976 HepG2 NT cells.
    assert len(controls) == 4976 and (store.fold[controls] == 0).all()
    targets = {name: np.flatnonzero(store.fold == i) for i, name in [(1, "validation"), (2, "test")]}
    cases = []
    for name, rows in targets.items():
        assert (lines[rows] == "HepG2").all() and not np.intersect1d(rows, controls).size
        cases.append(dict(key=name + "__HepG2", split=name, cell_line="HepG2",
            target_cells=len(rows), conditions=len(np.unique(store.condition_rows[rows])),
            reference_cells=len(controls)))
    assert [(c["target_cells"], c["conditions"]) for c in cases] == [(14091, 200), (28177, 372)]
    union = np.unique(np.concatenate([controls, *targets.values()]))
    assert len(union) == 47244
    return targets, controls, union, cases


def request_config(run):
    cfg = read_json(run / "request.json")
    cfg.setdefault("preset", "original-control1")
    cfg.setdefault("control_cells", 1)
    cfg.setdefault("seed", 420)
    return cfg


def evaluation_rows(store):
    # Public compatibility helper for the original-split acceptance check.
    return original_evaluation_rows(store)


def case_rows(cfg, store):
    if cfg["preset"].startswith("original-"):
        targets, controls, union, cases = original_evaluation_rows(store)
        return {c["key"]: (targets[c["split"]], controls) for c in cases}, union, cases
    # Frozen A/B case membership and per-line reference pools are shared with C.
    from replogle_control_eval_common import EVAL_ART, verify_inputs
    verify_inputs()
    plan = read_json(EVAL_ART / "plan.json")
    rows = {c["key"]: tuple(np.load(EVAL_ART / f"{c['key']}_{kind}.npy")
                            for kind in ("targets", "controls")) for c in plan["cases"]}
    union = np.unique(np.concatenate([v for pair in rows.values() for v in pair]))
    np.testing.assert_array_equal(union, np.load(EVAL_ART / "generation_rows.npy"))
    return rows, union, plan["cases"]


def eval_store(cfg):
    return CellStore(ART if cfg["preset"].startswith("original-") else AB_ART)


def input_control_rows(prefixes, rows, n):
    if n == 0:
        return np.empty((len(rows), 0), dtype=np.int64)
    if hasattr(prefixes, "control_indices"):
        values = np.stack([prefixes.control_indices(int(row)) for row in rows]) if len(rows) else np.empty((0, n), dtype=np.int64)
        return values[:, 0] if n == 1 else values
    return np.array([prefixes.control_index(int(row)) for row in rows], dtype=np.int64)


def batch_positions(n, batch_size, rank, world):
    if batch_size < 1 or world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid generation shard configuration")
    for bid in range(rank, math.ceil(n / batch_size), world):
        yield bid, np.arange(bid * batch_size, min((bid + 1) * batch_size, n), dtype=np.int64)


def prepare(run):
    request = request_config(run)
    verify_preset(request)
    output = run / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    formal = run / "formal"
    if request.get("mode") == "standalone_evaluation":
        from replogle_qwen_eval_request import verify_source
        verify_source(request)
        formal = Path(request["source_formal_dir"])
    done = read_json(formal / "completion.json")
    assert done["status"] == "complete" and done.get("phase", "formal") == "formal"
    model = formal / "final"
    selected = Path(done["best_checkpoint"]).resolve()
    assert selected.is_relative_to(formal.resolve())
    model_hash = sha(model / "model.safetensors")
    assert model_hash == sha(selected / "model.safetensors")
    assert model_hash == request.get("source_model_sha256", model_hash)
    if (output / "preparation_complete.json").exists():
        plan = read_json(output / "plan.json")
        assert plan["model_sha256"] == model_hash
        with h5py.File(output / "real_cells.h5", "r") as h:
            assert h.attrs["complete"] and len(h["global_rows"]) == plan["unique_generation_cells"]
        return
    store = eval_store(request)
    membership, rows, cases = case_rows(request, store)
    plan = dict(cases=cases, model=str(model), selected_checkpoint=str(selected), model_sha256=model_hash,
        source_formal_dir=str(formal), standalone_evaluation=request.get("mode") == "standalone_evaluation",
        source_artifacts=str(store.art), unique_generation_cells=len(rows), seed=request["seed"], preset=request["preset"],
        control_cells=request["control_cells"], eval_gpus=request["eval_gpus"], batch_size=request["eval_batch_size_generation"],
        sampling="One ancestral 64-code draw per row, temperature 1; seed=base_seed+global_batch_id; independent of GPU count at fixed batch size",
        nb_depth="Observed target 2000-HVG raw-count sum via original VAE.decode_observed",
        controls="Preset reference pools; prompt controls from same line/gem, excluding target self",
        metrics=list(METRICS), target_cell_cap=0, distribution_max_samples=5000, pca_components=30,
        scope="Original split" if request["preset"].startswith("original-") else "Stage-2 HepG2 holdout plus combination splits; NB tokenizer saw HepG2",
        validation_and_test_share_generated_reference=True)
    # Load raw data once on CPU, not independently in all eight generation workers.
    import draccus
    from sc_evae.config.training import TrainConfig
    from sc_evae.training.data_loader import PerturbationDataset
    with (ROOT / store.sources["vae_run"] / "config.yaml").open() as f:
        cfg = draccus.load(TrainConfig, f)
    cfg.dataset.pert_mapping_path = None
    assert not cfg.dataset.apply_normalize and not cfg.dataset.apply_log1p
    ds = PerturbationDataset(config=cfg.dataset)
    cache = ROOT / store.sources["nb_cache"]
    np.testing.assert_array_equal(ds._pert_indices, np.load(cache / "pert_idx.npy"))
    np.testing.assert_array_equal(ds._cell_type_indices, np.load(cache / "cell_type_idx.npy"))
    np.testing.assert_array_equal(ds._target_genes, (ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt").read_text().splitlines())
    counts = ds.counts[rows]
    np.testing.assert_array_equal(counts.sum(-1), np.load(cache / "hvg_library_size.npy")[rows])
    partial = output / "real_cells.partial.h5"
    with h5py.File(partial, "w") as h:
        h.attrs["complete"] = False
        h.create_dataset("global_rows", data=rows)
        h.create_dataset("true_counts", data=counts, dtype="float32", chunks=(64, 2000))
        h.create_dataset("pert_idx", data=ds._pert_indices[rows])
        h.create_dataset("cell_type_idx", data=ds._cell_type_indices[rows])
        for key, (targets, controls) in membership.items():
            h.create_dataset(key + "_targets", data=targets)
            h.create_dataset(key + "_controls", data=controls)
        h.attrs["complete"] = True
    partial.replace(output / "real_cells.h5")
    save_json(output / "plan.json", plan)
    save_json(output / "preparation_complete.json", dict(status="complete", cells=len(rows), model_sha256=model_hash))


def generate(run, batch_size):
    from transformers import AutoTokenizer, set_seed
    rank, local, world = (int(os.environ[k]) for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
    torch.cuda.set_device(local)
    output = run / "evaluation"
    plan = read_json(output / "plan.json")
    assert batch_size == plan["batch_size"] and world == plan["eval_gpus"]
    with h5py.File(output / "real_cells.h5", "r") as h:
        assert h.attrs["complete"]
        all_rows = h["global_rows"][:]
    batches = list(batch_positions(len(all_rows), batch_size, rank, world))
    positions = np.concatenate([pos for _, pos in batches]) if batches else np.empty(0, dtype=np.int64)
    rows = all_rows[positions]
    shards = output / "shards"
    shards.mkdir(exist_ok=True)
    path = shards / f"rank-{rank:02d}.h5"
    if path.exists():
        with h5py.File(path, "r") as h:
            assert h.attrs["complete"] and h.attrs["world_size"] == world
            assert h.attrs["model_sha256"] == plan["model_sha256"]
            np.testing.assert_array_equal(h["positions"][:], positions)
        return
    cfg = request_config(run)
    store = eval_store(cfg)
    model_dir = Path(plan["model"])
    install_fast_delta()
    set_seed(plan["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
    model = load_llm(model_dir, "cuda")
    vae = load_nb_decoder(ROOT / store.sources["vae_run"], "cuda")
    prefixes = make_dataset(cfg, DEFAULT_ART if cfg["preset"] == "original-text" else store, rows, tokenizer)
    started = time.monotonic()
    partial = path.with_suffix(".partial.h5")
    with h5py.File(output / "real_cells.h5", "r") as real, h5py.File(partial, "w") as h:
        h.attrs.update(complete=False, rank=rank, world_size=world, model_sha256=plan["model_sha256"])
        h.create_dataset("positions", data=positions)
        h.create_dataset("global_rows", data=rows)
        h.create_dataset("control_input_row", data=input_control_rows(prefixes, rows, plan["control_cells"]))
        codes_out = h.create_dataset("generated_codes", (len(rows), 64), dtype="uint16")
        predictions = h.create_dataset("predicted_counts", (len(rows), 2000), dtype="float32")
        offset = 0
        for bid, pos in batches:
            # Global batch IDs give distinct streams and do not depend on rank assignment.
            set_seed(plan["seed"] + bid)
            counts = torch.from_numpy(real["true_counts"][pos]).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                codes, pred = predict_cells(model, tokenizer, vae,
                    [prefixes.inference_prefix(i) for i in range(offset, offset + len(pos))], counts)
            assert torch.isfinite(pred).all() and (pred >= 0).all()
            codes_out[offset:offset+len(pos)] = codes.cpu().numpy().astype(np.uint16)
            predictions[offset:offset+len(pos)] = pred.cpu().numpy()
            offset += len(pos)
            if offset == len(pos) or bid % (world * 25) == rank or offset == len(rows):
                save_json(shards / f"rank-{rank:02d}-progress.json", dict(cells=offset, total=len(rows),
                    elapsed_seconds=time.monotonic()-started, gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
                print(f"GPU {rank}: {offset}/{len(rows)} cells", flush=True)
        assert offset == len(rows)
        h.attrs["complete"] = True
    partial.replace(path)


def merge_shards(output, plan):
    """Require complete, unique coverage before any population metric is computed."""
    with h5py.File(output / "real_cells.h5", "r") as real:
        rows = real["global_rows"][:]
    if (output / "generation_complete.json").exists():
        assert read_json(output / "generation_complete.json")["model_sha256"] == plan["model_sha256"]
        with h5py.File(output / "generated_cells.h5", "r") as h:
            assert h.attrs["complete"]
            np.testing.assert_array_equal(h["global_rows"][:], rows)
        return
    seen = np.zeros(len(rows), dtype=bool)
    partial = output / "generated_cells.partial.h5"
    with h5py.File(partial, "w") as out:
        out.attrs.update(complete=False, model_sha256=plan["model_sha256"])
        out.create_dataset("global_rows", data=rows)
        pred = out.create_dataset("predicted_counts", (len(rows), 2000), dtype="float32")
        codes = out.create_dataset("generated_codes", (len(rows), 64), dtype="uint16")
        n = plan.get("control_cells", 1)
        controls = out.create_dataset("control_input_row", (len(rows),) if n == 1 else (len(rows), n), dtype="int64")
        for rank in range(plan["eval_gpus"]):
            with h5py.File(output / "shards" / f"rank-{rank:02d}.h5", "r") as h:
                assert h.attrs["complete"] and h.attrs["world_size"] == plan["eval_gpus"]
                assert h.attrs["rank"] == rank and h.attrs["model_sha256"] == plan["model_sha256"]
                pos = h["positions"][:]
                pieces = [p for _, p in batch_positions(len(rows), plan["batch_size"], rank, plan["eval_gpus"])]
                expected = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
                np.testing.assert_array_equal(pos, expected)
                assert np.all(np.diff(pos) > 0) and not seen[pos].any()
                np.testing.assert_array_equal(h["global_rows"][:], rows[pos])
                values = h["predicted_counts"][:]; z = h["generated_codes"][:]
                assert np.isfinite(values).all() and (values >= 0).all() and (z <= 511).all()
                pred[pos] = values; codes[pos] = z
                control_values = h["control_input_row"][:]
                assert control_values.shape == ((len(pos),) if n == 1 else (len(pos), n))
                if n and len(pos):
                    controls[pos] = control_values
                seen[pos] = True
        assert seen.all(), "Missing generated cells"
        out.attrs["complete"] = True
    partial.replace(output / "generated_cells.h5")
    save_json(output / "generation_complete.json", dict(status="complete", cells=len(rows),
        gpu_workers=plan["eval_gpus"], per_gpu_batch=plan["batch_size"], model_sha256=plan["model_sha256"]))


def metrics(run, split):
    # Reuse the exact-pseudobulk adapter; all metric formulas stay in scripts/eval.py.
    from replogle_qwen_metrics import evaluate_case
    output = run / "evaluation"
    plan = read_json(output / "plan.json")
    assert read_json(output / "generation_complete.json")["status"] == "complete"
    case = next(c for c in plan["cases"] if c["key"] == split or c["split"] == split)
    with h5py.File(output / "real_cells.h5", "r") as h:
        rows, counts = h["global_rows"][:], h["true_counts"][:]
        pert, ctypes = h["pert_idx"][:], h["cell_type_idx"][:]
        if case["key"] + "_targets" in h:
            wanted = np.sort(np.concatenate([h[case["key"] + "_targets"][:], h[case["key"] + "_controls"][:]]))
        else:
            wanted = np.sort(np.concatenate([h[split + "_rows"][:], h["control_rows"][:]]))
    selected = np.searchsorted(rows, wanted)
    np.testing.assert_array_equal(rows[selected], wanted)
    with h5py.File(output / "generated_cells.h5", "r") as h:
        assert h.attrs["complete"] and h.attrs["model_sha256"] == plan["model_sha256"]
        np.testing.assert_array_equal(rows, h["global_rows"][:])
        predictions = h["predicted_counts"][:]
    source = read_json(Path(plan["source_artifacts"]) / "sources.json")
    labels = read_json(ROOT / source["nb_cache"] / "labels.json")
    genes = (ROOT / "assets/replogle_scldm2k_gene_order.ensembl.txt").read_text().splitlines()
    config = yaml.safe_load((ROOT / source["vae_run"] / "config.yaml").read_text())
    config["_vae_run"] = source["vae_run"]
    evaluate_case(case, output, counts, predictions, pert, ctypes, labels, genes, selected, config)


def summarize(run):
    output = run / "evaluation"
    plan = read_json(output / "plan.json")
    reports = []
    for case in plan["cases"]:
        report = output / "reports" / case["key"]
        assert read_json(report / "completion.json")["status"] == "complete"
        values = nine_metrics(read_json(report / "eval_results/summary_metrics_symbol_aligned.json"))
        assert set(values) == set(METRICS) and all(np.isfinite(v) for v in values.values())
        reports.append({**case, "nine_metrics": values})
    aggregates = {}
    for split in ("combination_validation", "combination_test"):
        group = [r for r in reports if r["split"] == split]
        if group:
            aggregates[split] = dict(aggregation="unweighted cell-line mean; per-line scores retained",
                nine_metrics={m: float(np.mean([r["nine_metrics"][m] for r in group])) for m in METRICS})
    save_json(output / "all_metrics.json", dict(reports=reports, protocol=plan, combination_macro_cell_line=aggregates))
    names = ["W2 ↓", "MMD2 ↓", "FD ↓", "PDS ↑", "PR-AUC ↑", "Sp-sig ↑", "Sp-LFC ↑", "ovl@N ↑", "P-delta ↑"]
    table = ["# Qwen NB-FSQ：九项群体指标", "",
        f"评测模型：{plan['selected_checkpoint']}。训练完成后按内部验证 CE 选择，不按测试指标选择。", "",
        "| 评测集 | " + " | ".join(names) + " |", "| " + " | ".join(["---"] * 10) + " |"]
    for r in reports:
        label = r["key"]
        table.append("| " + label + " | " + " | ".join(f"{r['nine_metrics'][m]:.6f}" for m in METRICS) + " |")
    (output / "results.md").write_text("\n".join(table) + "\n")
    save_json(output / "completion.json", dict(status="complete", reports=len(reports), all_nine_metrics_reported=True))
    save_json(run / "pipeline_completion.json", dict(status="complete",
        training=str(Path(plan["source_formal_dir"]) / "completion.json") if plan.get("standalone_evaluation") else "formal/completion.json",
        training_performed=not plan.get("standalone_evaluation", False),
        evaluation="evaluation/completion.json", results="evaluation/results.md", selected_checkpoint=plan["selected_checkpoint"]))


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--protocol", choices=["sharded", "legacy-control", "legacy-text"], default="sharded")
    protocol, argv = selector.parse_known_args(argv)
    p = argparse.ArgumentParser(description="Unified Qwen evaluation. --protocol sharded (default), legacy-control or legacy-text.")
    if protocol.protocol != "sharded":
        from replogle_qwen_legacy_eval import run_control, run_text
        if protocol.protocol == "legacy-control":
            p.add_argument("--variant", choices=["A", "B", "C"], required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--batch-size", type=int, default=64)
            run_control(p.parse_args(argv))
        else:
            p.add_argument("--model-dir", type=Path, required=True)
            p.add_argument("--output-dir", type=Path, required=True)
            p.add_argument("--split", choices=["quick", "validation", "test"], default="quick")
            p.add_argument("--batch-size", type=int, default=32)
            p.add_argument("--artifacts", type=Path, default=DEFAULT_ART)
            run_text(p.parse_args(argv))
        return
    p.add_argument("phase", choices=["prepare", "generate", "merge", "metrics", "summarize"])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--split", help="Case key, or validation/test for the original split")
    args = p.parse_args(argv)
    cfg = request_config(args.run_dir)
    if cfg.get("mode") == "standalone_evaluation":
        from replogle_qwen_eval_request import safe_output, verify_source
        run = safe_output(args.run_dir, cfg)
        verify_source(cfg)
    elif args.run_dir.resolve().is_relative_to((ROOT / "outputs/replogle_qwen").resolve()):
        run = preset_safe_output(args.run_dir, cfg)
    else:
        run = baseline_safe_output(args.run_dir)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if args.phase == "prepare": prepare(run)
    elif args.phase == "generate": generate(run, args.batch_size)
    elif args.phase == "merge": merge_shards(run / "evaluation", read_json(run / "evaluation/plan.json"))
    elif args.phase == "metrics":
        if not args.split: p.error("metrics requires --split")
        metrics(run, args.split)
    else: summarize(run)


if __name__ == "__main__":
    main()
