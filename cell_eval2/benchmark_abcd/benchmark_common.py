"""Shared measurement rules. Linux only; each repetition uses a new process.

Time includes input loading, preprocessing and scoring, but excludes imports,
CUDA initialization, optional warmup, dataset conversion and output writing.
RSS is the process lifetime high-water mark INCLUDING imports and warmup.
No cell-eval cache is enabled. OS filesystem caches are NOT cleared.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
DE_METRICS = [
    "de_wilcoxon_overlap_top100",
    "de_wilcoxon_sig_recall",
    "de_wilcoxon_direction_precision",
]


def main(case, route, device):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE / "data/prepared")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--with-de", action="store_true",
                        help="Add three DE metrics; defaults to separate results_de output")
    parser.add_argument("--cpu-de-materialize", action="store_true",
                        help="Explicitly allow C + DE to load the entire archive into RAM")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, choices=[0, 1], default=1)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cpu_de_materialize and not (args.with_de and route != "regular" and device == "cpu"):
        parser.error("--cpu-de-materialize is only valid for C with --with-de")
    if args.with_de and route != "regular" and device == "cpu" and not args.cpu_de_materialize:
        parser.error("Streaming DE requires CUDA. For C, explicitly add --cpu-de-materialize "
                     "to benchmark whole-archive loading followed by CPU DE.")
    if args.out is None:
        args.out = BASE / ("results_de" if args.with_de else "results")
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
    args.data = args.data.resolve()
    case_dir = args.out.resolve() / case
    case_dir.mkdir(parents=True, exist_ok=True)
    # Set BEFORE importing numpy / polars / cell_eval2, in all four schemes.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[key] = str(args.threads)
    if args.worker is not None:
        worker(case, route, device, args, case_dir)
        return
    reports = []
    # Sequential subprocesses avoid simultaneous CPU/GPU workloads.
    for repeat in range(1, args.repeats + 1):
        print(f"\n[{case}] 第 {repeat}/{args.repeats} 次：启动独立进程", flush=True)
        cmd = [sys.executable, "-u", str(Path(sys.argv[0]).resolve()),
               "--data", str(args.data), "--out", str(args.out.resolve()),
               "--threads", str(args.threads), "--warmup", str(args.warmup),
               "--worker", str(repeat)]
        if args.with_de:
            cmd.append("--with-de")
        if args.cpu_de_materialize:
            cmd.append("--cpu-de-materialize")
        log_path = case_dir / f"run_{repeat}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            launched = time.monotonic()
            last_notice = launched
            with log_path.open(encoding="utf-8", errors="replace") as reader:
                try:
                    while True:
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                        text = reader.read()
                        if text:
                            print(text, end="", flush=True)
                        if process.returncode is not None:
                            break
                        now = time.monotonic()
                        if now - last_notice >= 10:
                            print(f"[{case} {repeat}/{args.repeats}] 仍在运行，本轮已用 "
                                  f"{now-launched:.0f} 秒（含初始化、预热；不是正式计时）", flush=True)
                            last_notice = now
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
        if process.returncode:
            raise RuntimeError(f"{case} run {repeat} failed: {case_dir / f'run_{repeat}.log'}")
        report = json.loads((case_dir / f"run_{repeat}.json").read_text())
        reports.append(report)
        print(f"{case} repeat {repeat}: {report['seconds']:.3f} s, "
              f"process peak RSS {report['process_peak_rss_gib']:.3f} GiB", flush=True)
    with (case_dir / "timings.csv").open("w", newline="") as f:
        fields = ["case", "repeat", "seconds", "process_peak_rss_gib"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    print(f"{case} median: {statistics.median(r['seconds'] for r in reports):.3f} s")
    print(f"Results: {case_dir}")


def worker(case, route, device, args, case_dir):
    print(f"[{case}] 正在导入依赖、读取配置……", flush=True)
    import resource
    import polars as pl
    from cell_eval2 import EvalConfig, compute_metrics, aggregate_metrics
    from cell_eval2.config import DEParams
    from cell_eval2.catalog import resolve_metrics
    from cell_eval2.scale import score_streaming_cell

    manifest_path = args.data / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    # Retain the manifest's metrics; DE is an explicit additive selection.
    selected = list(manifest["metrics"])
    if args.with_de:
        selected = list(dict.fromkeys(selected + DE_METRICS))
    backend = "gpudge" if args.with_de and device == "cuda" else "pdex"
    evaluation_path = ("cellstream_materialized_cpu" if args.cpu_de_materialize
                       else "h5ad_in_memory" if route == "regular" else "cellstream_streaming")
    print(f"[{case}] metrics={selected}; DE backend={backend if args.with_de else 'disabled'}; "
          f"path={evaluation_path}", flush=True)
    if args.cpu_de_materialize:
        print(f"[{case}] C + DE: 整个 cellstream 文件读入内存后用 pdex；不是流式 CPU DE。", flush=True)
    cfg = EvalConfig(
        metrics=selected, pert_col="perturbation", control="control",
        input_type="counts", device=device, control_source="real",
        de=DEParams(backend=backend),
        gather_threads=args.threads,
        cache_real=None, cache_pred=None,
    )
    gpu_info = None
    if device == "cuda":
        print(f"[{case}] 正在初始化 GPU……", flush=True)
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device; refusing a silent CPU fallback")
        cp.asarray([1.0]).sum()  # Initialize context outside timed region.
        cp.cuda.Device().synchronize()
        if args.with_de:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("GPU DE requires torch CUDA; refusing CPU fallback")
            torch.cuda.init()
            torch.cuda.synchronize()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        gpu_info = props["name"]
        if isinstance(gpu_info, bytes):
            gpu_info = gpu_info.decode()

    suffix = ".h5ad" if route == "regular" else ".csad"
    pred, real = args.data / ("pred" + suffix), args.data / ("real" + suffix)
    for path in (pred, real):
        if not path.is_file():
            raise FileNotFoundError(path)

    def evaluate():
        if route == "regular" or args.cpu_de_materialize:
            # compute_metrics recognizes cell-layout archives and materializes them.
            return compute_metrics(str(pred), str(real), config=cfg)
        return score_streaming_cell(str(pred), str(real), config=cfg)

    def synchronize():
        if device == "cuda":
            cp.cuda.Device().synchronize()
            if args.with_de:
                torch.cuda.synchronize()

    for _ in range(args.warmup):
        print(f"[{case}] 预热评估开始（不计入正式耗时）……", flush=True)
        warm = evaluate()
        del warm
        synchronize()
    import gc
    gc.collect()
    synchronize()
    print(f"[{case}] 正式评估开始：读取数据、预处理、计算指标……", flush=True)
    start = time.perf_counter()
    result = evaluate()
    synchronize()
    seconds = time.perf_counter() - start
    print(f"[{case}] 正式评估结束：{seconds:.3f} 秒；正在检查和保存结果……", flush=True)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if result.is_empty():
        raise RuntimeError("Empty metric result")
    if result.select(pl.struct(["perturbation", "metric"]).n_unique()).item() != result.height:
        raise RuntimeError("Duplicate (perturbation, metric) keys")
    expected, missing = resolve_metrics(cfg.metrics, version=cfg.version)
    # Derived aggregate metrics have no per-perturbation row.
    from cell_eval2.catalog import CATALOG
    expected_raw = {m for m in expected if CATALOG[m].derived is None}
    if missing or set(result["metric"].unique()) != expected_raw:
        raise RuntimeError("Unexpected/missing metrics")
    result.sort(["perturbation", "metric"]).write_csv(case_dir / f"run_{args.worker}_results.csv")
    aggregate_metrics(result).write_csv(case_dir / f"run_{args.worker}_agg.csv")
    cfg.to_yaml(str(case_dir / "config.yaml"))
    packages = {}
    for name in ("cell-eval2", "cellstream", "numpy", "scipy", "anndata", "polars", "cupy-cuda12x",
                 "pdex", "gpudge", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = dict(case=case, route=route, device=device, repeat=args.worker,
                  with_de=args.with_de, de_backend=backend if args.with_de else None,
                  evaluation_path=evaluation_path,
                  seconds=seconds, process_peak_rss_gib=peak_rss,
                  warmup=args.warmup, threads=args.threads, gpu=gpu_info,
                  python=sys.version, platform=platform.platform(), packages=packages,
                  dataset_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  metrics=cfg.metrics, n_cells=manifest["n_cells"], n_genes=manifest["n_genes"],
                  n_perturbations=manifest["n_perturbations"],
                  input_bytes={str(p.name): p.stat().st_size for p in (pred, real)},
                  timing_scope="load+prepare+score; no conversion/imports/warmup/output",
                  rss_scope="process lifetime high-water including imports and warmup",
                  filesystem_cache="not cleared", cell_eval_cache=False)
    (case_dir / f"run_{args.worker}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

