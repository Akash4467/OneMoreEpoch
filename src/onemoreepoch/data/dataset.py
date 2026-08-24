"""Dataset: the minimal indexable-and-sized interface DataLoader batches over."""

from abc import ABC, abstractmethod
from typing import Any

from onemoreepoch.exceptions import DataError


class Dataset(ABC):
    """Anything sized and indexable by integer position."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Any: ...


class TensorDataset(Dataset):
    """Wraps parallel arrays (e.g. features, labels) as a Dataset of tuples."""

    def __init__(self, *arrays: Any) -> None:
        if not arrays:
            raise DataError("dataset_empty")
        length = len(arrays[0])
        if any(len(a) != length for a in arrays):
            raise DataError("dataset_length_mismatch", lengths=[len(a) for a in arrays])
        self._arrays = arrays
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple:
        return tuple(a[index] for a in self._arrays)
