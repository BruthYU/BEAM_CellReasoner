"""Sampling policies for large collections of per-study CellStream archives."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class ArchiveBatchSampler(Sampler[list[int]]):
    """Yield shuffled batches whose rows all belong to one archive.

    A globally shuffled cell sampler scatters almost every batch across many
    study archives. That defeats CellStream's batched ``gather_rows`` API and
    causes hundreds of tiny reads. This sampler shuffles archive order and the
    rows within each archive while keeping each emitted batch archive-local.

    In distributed runs every rank constructs the same global batch sequence
    and consumes a disjoint strided subset. The tail is truncated so all ranks
    execute the same number of optimizer steps.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.lengths = tuple(int(length) for length in lengths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if not self.lengths or any(length < 0 for length in self.lengths):
            raise ValueError("archive lengths must be a non-empty non-negative sequence")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.offsets = tuple(np.cumsum([0, *self.lengths], dtype=np.int64).tolist())
        if self.drop_last:
            total_batches = sum(length // self.batch_size for length in self.lengths)
        else:
            total_batches = sum(
                math.ceil(length / self.batch_size) for length in self.lengths if length
            )
        self.global_batches = total_batches - (total_batches % self.num_replicas)
        self.rank_batches = self.global_batches // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.rank_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        archive_order = np.arange(len(self.lengths), dtype=np.int64)
        if self.shuffle:
            rng.shuffle(archive_order)

        global_batch = 0
        for archive_index_value in archive_order:
            archive_index = int(archive_index_value)
            length = self.lengths[archive_index]
            rows = (
                rng.permutation(length)
                if self.shuffle
                else np.arange(length, dtype=np.int64)
            )
            stop = length if not self.drop_last else length - (length % self.batch_size)
            for start in range(0, stop, self.batch_size):
                if global_batch >= self.global_batches:
                    return
                batch = rows[start : min(start + self.batch_size, length)]
                if global_batch % self.num_replicas == self.rank:
                    offset = self.offsets[archive_index]
                    yield (batch + offset).tolist()
                global_batch += 1
