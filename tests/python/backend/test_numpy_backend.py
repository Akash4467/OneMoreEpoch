"""Tests for the backend abstraction and NumPy implementation."""

import numpy as np
import pytest

from onemoreepoch.core.backend import NumPyBackend, get_backend, register_backend
from onemoreepoch.exceptions import BackendError


@pytest.fixture
def backend():
    return NumPyBackend()


class TestRegistry:
    def test_default_backend_is_numpy(self):
        assert get_backend().name == "numpy"

    def test_lookup_by_name(self):
        assert get_backend("numpy").name == "numpy"

    def test_unknown_backend_raises(self):
        with pytest.raises(BackendError):
            get_backend("tpu")

    def test_register_custom_backend(self):
        custom = NumPyBackend()
        custom.name = "custom"
        register_backend(custom)
        assert get_backend("custom") is custom


class TestCreation:
    def test_array(self, backend):
        arr = backend.array([1, 2, 3])
        assert isinstance(arr, np.ndarray)

    def test_zeros_ones_full(self, backend):
        assert backend.zeros((2, 3)).sum() == 0
        assert backend.ones((2, 3)).sum() == 6
        assert backend.full((2, 2), 7).sum() == 28

    def test_like_variants(self, backend):
        base = backend.array([[1.0, 2.0]])
        assert backend.zeros_like(base).shape == (1, 2)
        assert backend.ones_like(base).sum() == 2

    def test_is_native(self, backend):
        assert backend.is_native(np.array([1]))
        assert not backend.is_native([1])


class TestRandom:
    def test_seed_reproducibility(self, backend):
        backend.seed(42)
        first = backend.randn((3,))
        backend.seed(42)
        second = backend.randn((3,))
        np.testing.assert_array_equal(first, second)

    def test_rand_range(self, backend):
        sample = backend.rand((100,))
        assert (sample >= 0).all() and (sample < 1).all()


class TestOps:
    def test_arithmetic(self, backend):
        a, b = backend.array([2.0, 4.0]), backend.array([1.0, 2.0])
        np.testing.assert_array_equal(backend.add(a, b), [3, 6])
        np.testing.assert_array_equal(backend.subtract(a, b), [1, 2])
        np.testing.assert_array_equal(backend.multiply(a, b), [2, 8])
        np.testing.assert_array_equal(backend.divide(a, b), [2, 2])
        np.testing.assert_array_equal(backend.negative(a), [-2, -4])
        np.testing.assert_array_equal(backend.power(a, 2), [4, 16])

    def test_matmul(self, backend):
        a = backend.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.matmul(a, a)
        np.testing.assert_array_equal(result, [[7, 10], [15, 22]])

    def test_reductions(self, backend):
        a = backend.array([[1.0, 2.0], [3.0, 4.0]])
        assert backend.sum(a) == 10
        assert backend.mean(a) == 2.5
        assert backend.max(a) == 4
        np.testing.assert_array_equal(backend.sum(a, axis=0), [4, 6])

    def test_shape_ops(self, backend):
        a = backend.array([[1.0, 2.0], [3.0, 4.0]])
        assert backend.reshape(a, (4,)).shape == (4,)
        assert backend.transpose(a).shape == (2, 2)
        assert backend.broadcast_to(backend.array([1.0]), (3, 1)).shape == (3, 1)
