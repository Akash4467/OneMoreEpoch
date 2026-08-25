from abc import ABC, abstractmethod
from typing import Any

from onemoreepoch.exceptions import DataError


# Minimal indexable-and-sized interface DataLoader batches over
class Dataset(ABC):
    # Returns the number of samples
    @abstractmethod
    def __len__(self) -> int: ...

    # Returns the sample at index
    @abstractmethod
    def __getitem__(self, index: int) -> Any: ...


# Wraps parallel arrays (e.g. features, labels) as a Dataset of tuples
class TensorDataset(Dataset):
    # Validates the arrays are non-empty and equal length
    def __init__(self, *arrays: Any) -> None:
        if not arrays:
            raise DataError("dataset_empty")
        length = len(arrays[0])
        if any(len(a) != length for a in arrays):
            raise DataError("dataset_length_mismatch", lengths=[len(a) for a in arrays])
        self._arrays = arrays
        self._length = length

    # Returns the number of samples
    def __len__(self) -> int:
        return self._length

    # Returns the tuple of per-array values at index
    def __getitem__(self, index: int) -> tuple:
        return tuple(a[index] for a in self._arrays)
