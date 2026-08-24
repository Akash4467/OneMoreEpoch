"""DataLoader: batches and (optionally) shuffles samples from a Dataset.

Collation happens on raw Python/NumPy data, before anything becomes a
Tensor — this is the "Input/output interoperability... easy data
manipulation" role the doc reserves for NumPy (§10), not a backend
circumvention.
"""

from collections.abc import Iterator

import numpy as np

from onemoreepoch.data.dataset import Dataset
from onemoreepoch.exceptions import DataError


class DataLoader:
    """Iterates a Dataset in batches, optionally shuffled each pass."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        *,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        if len(dataset) == 0:
            raise DataError("dataset_empty")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise DataError("dataloader_bad_batch_size", value=batch_size)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple]:
        order = np.arange(len(self.dataset))
        if self.shuffle:
            self._rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            samples = [self.dataset[int(i)] for i in indices]
            yield _collate(samples)


def _collate(samples: list) -> tuple | np.ndarray:
    first = samples[0]
    if isinstance(first, tuple):
        return tuple(
            np.stack([sample[i] for sample in samples]) for i in range(len(first))
        )
    return np.stack(samples)
