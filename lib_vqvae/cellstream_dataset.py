"""CellStream-backed datasets and CellTok expression preprocessing.

The archives contain sparse expression values aligned to the exact gene order
in ``data/5000_gene_names.csv``. They may store raw integer counts (recommended
for full training) or source ``X`` float values (needed to reproduce a released
tutorial). Normalization is deliberately performed after batch reads.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from lib_vqvae.paths import resolve_data_path


def normalize_expression_batch(
    matrix,
    *,
    target_sum: float = 1e4,
    log_transform: str = "log1p",
) -> np.ndarray:
    """Library-size normalize a sparse/dense batch and return dense float32.

    ``log1p`` matches the released CellTok notebooks. ``log10`` matches equation
    (1) in the paper and is kept as an explicit ablation option.
    """

    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    if log_transform not in {"log1p", "log10", "none"}:
        raise ValueError("log_transform must be one of: log1p, log10, none")

    if sp.issparse(matrix):
        values = matrix.tocsr().astype(np.float32, copy=True)
        row_sums = np.asarray(values.sum(axis=1)).reshape(-1)
        scales = np.divide(
            np.float32(target_sum),
            row_sums,
            out=np.zeros_like(row_sums, dtype=np.float32),
            where=row_sums > 0,
        )
        values = sp.diags(scales, format="csr") @ values
        if log_transform == "log1p":
            values.data = np.log1p(values.data)
        elif log_transform == "log10":
            values.data = np.log10(1.0 + values.data)
        return values.toarray().astype(np.float32, copy=False)

    values = np.asarray(matrix, dtype=np.float32).copy()
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D expression batch, got shape {values.shape}")
    row_sums = values.sum(axis=1)
    scales = np.divide(
        np.float32(target_sum),
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float32),
        where=row_sums > 0,
    )
    values *= scales[:, None]
    if log_transform == "log1p":
        np.log1p(values, out=values)
    elif log_transform == "log10":
        np.log10(1.0 + values, out=values)
    return values


def read_archive_manifest(path: str | Path, *, split: str | None = None) -> list[Path]:
    """Read archive paths from a CSV manifest.

    Supported path columns are ``cellstream_path`` and ``path``. Relative paths
    are resolved relative to the manifest, not the caller's working directory.
    """

    manifest_path = resolve_data_path(path)
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest contains no rows: {manifest_path}")

    archives: list[Path] = []
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        value = row.get("cellstream_path") or row.get("path")
        if not value:
            raise ValueError(
                f"manifest row is missing cellstream_path/path: {manifest_path}"
            )
        archives.append(resolve_data_path(value, base_dir=manifest_path.parent))
    if not archives:
        suffix = f" for split={split!r}" if split is not None else ""
        raise ValueError(f"manifest contains no archives{suffix}: {manifest_path}")
    return archives


def stage_archives_locally(
    archives: Sequence[str | Path], local_cache_dir: str | Path
) -> list[Path]:
    """Copy packed archives to an mmap-capable node-local filesystem.

    The cluster's GPFS mount is exposed through virtiofs, which rejects mmap.
    CellStream's packed reader requires mmap, so each selected per-study archive
    must be staged locally before it is opened. A POSIX lock prevents duplicate
    copies when several DDP ranks on the same node start together.
    """

    import fcntl

    sources = [resolve_data_path(path) for path in archives]
    cache_dir = Path(local_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    required = 0
    for source in sources:
        prefix = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        destination = cache_dir / f"{prefix}-{source.name}"
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            required += source.stat().st_size
    free = shutil.disk_usage(cache_dir).free
    if required > free:
        raise OSError(
            f"CellStream local cache needs {required} bytes but {cache_dir} has {free} free"
        )

    staged: list[Path] = []
    for source in sources:
        prefix = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        destination = cache_dir / f"{prefix}-{source.name}"
        lock_path = destination.with_suffix(destination.suffix + ".lock")
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
                temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.partial")
                shutil.copyfile(source, temporary)
                if temporary.stat().st_size != source.stat().st_size:
                    raise OSError(f"incomplete CellStream staging copy: {source}")
                os.replace(temporary, destination)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        staged.append(destination)
    return staged


class CellStreamDataset(Dataset):
    """Map-style PyTorch dataset over one or more CellStream archives.

    PyTorch calls ``__getitems__`` with a full batch of indices.  We exploit that
    hook to group the request by archive and issue one CellStream ``gather_rows``
    call per archive rather than decoding cells one at a time.
    """

    def __init__(
        self,
        archives: Sequence[str | Path],
        *,
        expected_genes: int = 5000,
        target_sum: float = 1e4,
        log_transform: str = "log1p",
        read_threads: int = 1,
        max_open_archives: int | None = None,
    ) -> None:
        if not archives:
            raise ValueError("at least one CellStream archive is required")
        if read_threads < 1:
            raise ValueError("read_threads must be >= 1")
        if max_open_archives is not None and max_open_archives < 1:
            raise ValueError("max_open_archives must be >= 1")

        try:
            import cellstream
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "CellStream is required; install cellstream==0.10.0 under Python >=3.11"
            ) from exc

        self.archives = tuple(str(resolve_data_path(path)) for path in archives)
        self.expected_genes = int(expected_genes)
        self.target_sum = float(target_sum)
        self.log_transform = log_transform
        self.read_threads = int(read_threads)
        self.max_open_archives = max_open_archives
        self._stores: dict[int, object] = {}

        lengths: list[int] = []
        for path in self.archives:
            store = cellstream.open(path)
            try:
                n_obs, n_vars = store.shape
                if n_vars != self.expected_genes:
                    raise ValueError(
                        f"{path} contains {n_vars} genes; expected {self.expected_genes}"
                    )
                lengths.append(int(n_obs))
            finally:
                store.close()
        self.lengths = tuple(lengths)
        self.offsets = tuple(np.cumsum([0, *lengths], dtype=np.int64).tolist())

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path,
        *,
        split: str | None = None,
        local_cache_dir: str | Path | None = None,
        **kwargs,
    ) -> "CellStreamDataset":
        archives = read_archive_manifest(manifest, split=split)
        if local_cache_dir is not None:
            archives = stage_archives_locally(archives, local_cache_dir)
        return cls(archives, **kwargs)

    def __len__(self) -> int:
        return self.offsets[-1]

    def _store(self, archive_index: int):
        if archive_index not in self._stores:
            import cellstream

            if self.max_open_archives is not None:
                while len(self._stores) >= self.max_open_archives:
                    oldest = next(iter(self._stores))
                    self._stores.pop(oldest).close()
            self._stores[archive_index] = cellstream.open(self.archives[archive_index])
        return self._stores[archive_index]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        archive_index = bisect.bisect_right(self.offsets, index) - 1
        return archive_index, index - self.offsets[archive_index]

    def get_batch_numpy(self, indices: Sequence[int]) -> np.ndarray:
        """Read and normalize an arbitrary batch while preserving input order."""

        normalized_indices = [int(index) for index in indices]
        if not normalized_indices:
            return np.empty((0, self.expected_genes), dtype=np.float32)

        requests: dict[int, list[tuple[int, int]]] = {}
        for output_position, index in enumerate(normalized_indices):
            archive_index, local_index = self._locate(index)
            requests.setdefault(archive_index, []).append((output_position, local_index))

        output = np.empty(
            (len(normalized_indices), self.expected_genes), dtype=np.float32
        )
        for archive_index, requested in requests.items():
            local_rows = [local_index for _, local_index in requested]
            matrix = self._store(archive_index).gather_rows(
                local_rows, n_threads=self.read_threads
            )
            batch = normalize_expression_batch(
                matrix,
                target_sum=self.target_sum,
                log_transform=self.log_transform,
            )
            for source_position, (output_position, _) in enumerate(requested):
                output[output_position] = batch[source_position]
        return output

    def get_obs_values(self, indices: Sequence[int], column: str) -> np.ndarray:
        """Return one observation column for arbitrary global row indices."""

        values = np.empty(len(indices), dtype=object)
        requests: dict[int, list[tuple[int, int]]] = {}
        for output_position, index in enumerate(indices):
            archive_index, local_index = self._locate(int(index))
            requests.setdefault(archive_index, []).append((output_position, local_index))
        for archive_index, requested in requests.items():
            obs = self._store(archive_index).obs
            if column not in obs:
                raise KeyError(f"{column!r} not found in {self.archives[archive_index]} obs")
            local_rows = [local_index for _, local_index in requested]
            selected = obs.iloc[local_rows][column].astype(str).to_numpy()
            for source_position, (output_position, _) in enumerate(requested):
                values[output_position] = selected[source_position]
        return values

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.get_batch_numpy([index])[0])

    def __getitems__(self, indices: Iterable[int]) -> list[torch.Tensor]:
        batch = self.get_batch_numpy(list(indices))
        return [torch.from_numpy(row) for row in batch]

    def close(self) -> None:
        for store in self._stores.values():
            try:
                store.close()
            except (BufferError, OSError):
                pass
        self._stores.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stores"] = {}
        return state

    def __del__(self):  # pragma: no cover - interpreter shutdown is non-deterministic
        self.close()
