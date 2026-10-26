"""B: h5ad -> compute_metrics, GPU. Run: python B_regular_gpu.py"""
from benchmark_common import main

if __name__ == "__main__":
    main("B", "regular", "cuda")
