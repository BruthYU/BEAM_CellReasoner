"""Summarize timings and compare ALL repetitions against A run 1 by row keys."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent


def read_values(path, aggregate=False):
    result = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = row["metric"] if aggregate else (row["perturbation"], row["metric"])
            if key in result:
                raise ValueError(f"Duplicate key {key} in {path}")
            value = row["mean" if aggregate else "value"]
            result[key] = float(value) if value else float("nan")
    return result


def compare(left, right, atol, rtol):
    if left.keys() != right.keys():
        return dict(keys_match=False, finite_max_abs_diff=None, outside_tolerance=None,
                    nonfinite_mismatches=None, matched_nonfinite=None, passed=False)
    diffs, bad, nonfinite_bad, matched_nonfinite = [], 0, 0, 0
    for key, a in left.items():
        b = right[key]
        if not math.isfinite(a) or not math.isfinite(b):
            equal = (math.isnan(a) and math.isnan(b)) or a == b
            nonfinite_bad += not equal
            matched_nonfinite += equal
        else:
            diffs.append(abs(a-b))
            bad += abs(a-b) > atol + rtol*abs(a)
    return dict(keys_match=True, finite_max_abs_diff=max(diffs, default=0),
                outside_tolerance=bad, nonfinite_mismatches=nonfinite_bad,
                matched_nonfinite=matched_nonfinite, passed=bad == 0 and nonfinite_bad == 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "results")
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    root = args.out.resolve()
    reports = {case: [json.loads(p.read_text()) for p in sorted((root / case).glob("run_*.json"))]
               for case in "ABCD"}
    if any(not records for records in reports.values()):
        raise RuntimeError("Need successful A, B, C, D runs before summarizing")
    reference_meta = reports["A"][0]
    base = read_values(root / "A/run_1_results.csv")
    base_agg = read_values(root / "A/run_1_agg.csv", True)
    timing_summary, consistency = [], []
    a_median = statistics.median(r["seconds"] for r in reports["A"])
    for case, records in reports.items():
        for record in records:
            for field in ("dataset_manifest_sha256", "metrics", "threads", "warmup", "packages"):
                if record[field] != reference_meta[field]:
                    raise ValueError(f"Incomparable runs: {case} has different {field}")
            repeat = record["repeat"]
            for kind, baseline, tail, is_agg in [("per_pert", base, "results", False),
                                                 ("aggregate", base_agg, "agg", True)]:
                result = compare(baseline, read_values(root / case / f"run_{repeat}_{tail}.csv", is_agg),
                                 args.atol, args.rtol)
                consistency.append(dict(case=case, repeat=repeat, kind=kind, **result))
        seconds = [r["seconds"] for r in records]
        med = statistics.median(seconds)
        timing_summary.append(dict(case=case, repeats=len(records), median_seconds=med,
                                   evaluation_path=records[0].get("evaluation_path", records[0]["route"]),
                                   de_backend=records[0].get("de_backend"),
                                   min_seconds=min(seconds), max_seconds=max(seconds),
                                   speedup_vs_A=a_median/med,
                                   max_process_peak_rss_gib=max(r["process_peak_rss_gib"] for r in records)))
    for filename, rows in [("summary.csv", timing_summary), ("consistency.csv", consistency)]:
        with (root / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("case | repeats | median seconds | A/time | process peak RSS GiB")
    for row in timing_summary:
        print(f"{row['case']} | {row['repeats']} | {row['median_seconds']:.3f} | "
              f"{row['speedup_vs_A']:.3f} | {row['max_process_peak_rss_gib']:.3f}")
    passed = all(row["passed"] for row in consistency)
    print(f"Numerical agreement: {passed}; atol={args.atol}, rtol={args.rtol}")
    print("Matching NaNs are reported separately; agreement does not imply every metric is defined.")
    if not passed:
        print("Inspect consistency.csv before claiming equivalent scoring.")
    if reference_meta.get("with_de"):
        print("DE: A/C use pdex, B/D use gpudge. Backend differences can change DE results.")
        print("C + DE materializes the cellstream archive in RAM; it is NOT CPU streaming DE.")
        print("Timings include the selected DE metrics, not all metrics or a separate DE-only timer.")
    print("CSV files:", root)
