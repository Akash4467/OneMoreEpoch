import numpy as np
import pytest

from onemoreepoch.data import DataLoader, TensorDataset
from onemoreepoch.exceptions import DataError


# Tests TensorDataset
class TestTensorDataset:
    # Checks length and item access return the right values
    def test_len_and_getitem(self):
        x = np.arange(10).reshape(5, 2)
        y = np.arange(5)
        ds = TensorDataset(x, y)
        assert len(ds) == 5
        xi, yi = ds[2]
        np.testing.assert_array_equal(xi, x[2])
        assert yi == y[2]

    # Checks constructing with no arrays raises
    def test_rejects_empty(self):
        with pytest.raises(DataError):
            TensorDataset()

    # Checks mismatched array lengths raise
    def test_rejects_mismatched_lengths(self):
        with pytest.raises(DataError):
            TensorDataset(np.zeros(4), np.zeros(5))


# Tests DataLoader
class TestDataLoader:
    # Checks every sample is seen exactly once across all batches
    def test_batches_all_samples(self):
        x = np.arange(10)
        loader = DataLoader(TensorDataset(x), batch_size=3)
        seen = []
        for (batch_x,) in loader:
            seen.extend(batch_x.tolist())
        assert sorted(seen) == list(range(10))

    # Checks batch shapes, including a short final batch
    def test_batch_shapes(self):
        x = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        batches = list(loader)
        assert batches[0][0].shape == (4, 2)
        assert batches[0][1].shape == (4,)
        assert batches[-1][0].shape == (2, 2)

    # Checks len() matches the ceiling-division batch count
    def test_len_matches_batch_count(self):
        loader = DataLoader(TensorDataset(np.arange(10)), batch_size=3)
        assert len(loader) == 4

    # Checks shuffling still covers every index
    def test_shuffle_covers_all_indices(self):
        x = np.arange(10)
        loader = DataLoader(TensorDataset(x), batch_size=10, shuffle=True, seed=0)
        (batch,) = next(iter(loader))
        assert sorted(batch.tolist()) == list(range(10))

    # Checks an empty dataset raises
    def test_rejects_empty_dataset(self):
        with pytest.raises(DataError):
            DataLoader(TensorDataset(np.arange(0)))

    # Checks a non-positive batch_size raises
    def test_rejects_bad_batch_size(self):
        with pytest.raises(DataError):
            DataLoader(TensorDataset(np.arange(5)), batch_size=0)
