"""Reconstruction and clustering metrics for CellTok stage 1."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
)


def reconstruction_metrics(original, reconstructed) -> dict[str, float]:
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if original.shape != reconstructed.shape or original.ndim != 2:
        raise ValueError(
            f"expected matching 2-D matrices, got {original.shape} and {reconstructed.shape}"
        )
    mse = float(np.mean(np.square(original - reconstructed)))

    original_centered = original - original.mean(axis=1, keepdims=True)
    reconstructed_centered = reconstructed - reconstructed.mean(axis=1, keepdims=True)
    pcc_denominator = np.linalg.norm(original_centered, axis=1) * np.linalg.norm(
        reconstructed_centered, axis=1
    )
    pcc = np.divide(
        np.sum(original_centered * reconstructed_centered, axis=1),
        pcc_denominator,
        out=np.zeros(original.shape[0], dtype=np.float64),
        where=pcc_denominator > 0,
    )

    cosine_denominator = np.linalg.norm(original, axis=1) * np.linalg.norm(
        reconstructed, axis=1
    )
    cosine = np.divide(
        np.sum(original * reconstructed, axis=1),
        cosine_denominator,
        out=np.zeros(original.shape[0], dtype=np.float64),
        where=cosine_denominator > 0,
    )
    gene_cosine_denominator = np.linalg.norm(original, axis=0) * np.linalg.norm(
        reconstructed, axis=0
    )
    gene_cosine = np.divide(
        np.sum(original * reconstructed, axis=0),
        gene_cosine_denominator,
        out=np.zeros(original.shape[1], dtype=np.float64),
        where=gene_cosine_denominator > 0,
    )
    global_cosine_denominator = np.linalg.norm(original) * np.linalg.norm(reconstructed)
    global_cosine = (
        float(np.sum(original * reconstructed) / global_cosine_denominator)
        if global_cosine_denominator > 0
        else 0.0
    )
    return {
        "mse": mse,
        "pcc": float(pcc.mean()),
        # The paper does not state whether vector pairs are cells or genes.
        # Keep `acs` backward compatible and report both interpretations.
        "acs": float(cosine.mean()),
        "acs_cellwise": float(cosine.mean()),
        "acs_genewise": float(gene_cosine.mean()),
        "acs_global": global_cosine,
    }


def codebook_usage_metrics(indices, *, n_codes: int) -> dict[str, object]:
    """Summarize discrete-code usage over cells and token positions.

    ``indices`` must have shape ``(cells, positions)``.  The returned scalar
    fields mirror the epoch-level diagnostics emitted by the trainer, while
    the per-position and per-code arrays make retrained/released checkpoint
    comparisons auditable on exactly the same evaluation cells.
    """

    values = np.asarray(indices)
    if values.ndim != 2:
        raise ValueError(f"code indices must be 2-D, got shape {values.shape}")
    if n_codes < 1:
        raise ValueError("n_codes must be positive")
    if values.size:
        if not np.issubdtype(values.dtype, np.integer):
            if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
                raise ValueError("code indices must be finite integers")
        values = values.astype(np.int64, copy=False)
        if values.min() < 0 or values.max() >= n_codes:
            raise ValueError(f"code indices must be in [0, {n_codes})")
    else:
        values = values.astype(np.int64, copy=False)

    positions = values.shape[1]
    counts = np.zeros((positions, n_codes), dtype=np.int64)
    for position in range(positions):
        counts[position] = np.bincount(values[:, position], minlength=n_codes)

    def perplexity(row: np.ndarray) -> float:
        total = int(row.sum())
        if total == 0:
            return 0.0
        probabilities = row[row > 0].astype(np.float64) / total
        return float(np.exp(-np.sum(probabilities * np.log(probabilities))))

    overall = counts.sum(axis=0)
    active_by_position = np.count_nonzero(counts, axis=1)
    perplexity_by_position = np.asarray(
        [perplexity(row) for row in counts], dtype=np.float64
    )
    active_codes = int(np.count_nonzero(overall))
    return {
        "cells": int(values.shape[0]),
        "positions": int(positions),
        "token_count": int(values.size),
        "active_codes": active_codes,
        "code_utilization": active_codes / n_codes,
        "code_perplexity": perplexity(overall),
        "active_codes_per_position_min": int(active_by_position.min()) if positions else 0,
        "active_codes_per_position_mean": float(active_by_position.mean()) if positions else 0.0,
        "active_codes_per_position_max": int(active_by_position.max()) if positions else 0,
        "perplexity_per_position_min": float(perplexity_by_position.min()) if positions else 0.0,
        "perplexity_per_position_mean": float(perplexity_by_position.mean()) if positions else 0.0,
        "perplexity_per_position_max": float(perplexity_by_position.max()) if positions else 0.0,
        "active_codes_by_position": active_by_position.tolist(),
        "perplexity_by_position": perplexity_by_position.tolist(),
        "counts_by_code": overall.tolist(),
    }


def _rbf_mean(x, y, gamma: float, block_size: int = 256) -> float:
    total = 0.0
    count = 0
    for start in range(0, x.shape[0], block_size):
        distances = pairwise_distances(x[start : start + block_size], y, metric="sqeuclidean")
        values = np.exp(-gamma * distances)
        total += float(values.sum())
        count += values.size
    return total / max(count, 1)


def rbf_mmd(
    original,
    reconstructed,
    *,
    max_cells: int = 2000,
    bandwidth_sample: int = 500,
    seed: int = 2024,
) -> tuple[float, float]:
    """Biased RBF MMD^2 with a reproducible median-distance bandwidth."""

    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    if original.ndim != 2 or original.shape[1:] != reconstructed.shape[1:]:
        raise ValueError("MMD inputs must be 2-D and have the same feature count")
    rng = np.random.default_rng(seed)
    if original.shape[0] > max_cells:
        original = original[rng.choice(original.shape[0], max_cells, replace=False)]
    if reconstructed.shape[0] > max_cells:
        reconstructed = reconstructed[
            rng.choice(reconstructed.shape[0], max_cells, replace=False)
        ]
    combined = np.concatenate([original, reconstructed], axis=0)
    if combined.shape[0] > bandwidth_sample:
        combined = combined[
            rng.choice(combined.shape[0], bandwidth_sample, replace=False)
        ]
    squared_distances = pairwise_distances(combined, metric="sqeuclidean")
    positive = squared_distances[squared_distances > 0]
    median_squared_distance = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / max(2.0 * median_squared_distance, np.finfo(float).eps)
    mmd2 = (
        _rbf_mean(original, original, gamma)
        + _rbf_mean(reconstructed, reconstructed, gamma)
        - 2.0 * _rbf_mean(original, reconstructed, gamma)
    )
    return float(max(mmd2, 0.0)), median_squared_distance


def clustering_metrics(
    representation,
    labels,
    *,
    pca_components: int | None = None,
    seed: int = 2024,
) -> dict[str, float]:
    representation = np.asarray(representation, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    if representation.ndim != 2 or representation.shape[0] != labels.shape[0]:
        raise ValueError("representation rows must match labels")
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("at least two ground-truth classes are required")

    features = representation
    if pca_components is not None:
        components = min(
            int(pca_components),
            features.shape[1],
            max(1, features.shape[0] - 1),
        )
        if components < features.shape[1]:
            features = PCA(n_components=components, random_state=seed).fit_transform(features)
    predicted = KMeans(
        n_clusters=unique_labels.size,
        random_state=seed,
        n_init=10,
    ).fit_predict(features)
    return {
        "ari": float(adjusted_rand_score(labels, predicted)),
        "nmi": float(normalized_mutual_info_score(labels, predicted)),
        "ami": float(adjusted_mutual_info_score(labels, predicted)),
    }
