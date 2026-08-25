from typing import Any

import numpy as np

from onemoreepoch.core.backend.base import Backend
from onemoreepoch.core.backend.factory import register_backend_factory
from onemoreepoch.core.backend.numpy_backend import NumPyBackend
from onemoreepoch.exceptions import BackendError

try:
    from onemoreepoch import _rustcore
except ImportError as exc:
    _rustcore = None
    _import_error = exc
else:
    _import_error = None


# Normalizes arbitrary input (list/tuple/scalar/ndarray/RustArray) into a RustArray
def _to_rust_array(data: Any) -> Any:
    if _rustcore is not None and isinstance(data, _rustcore.RustArray):
        return data
    array = np.asarray(data, dtype=np.float64)
    return _rustcore.RustArray.from_flat(array.ravel().tolist(), list(array.shape))


# Normalizes an axis argument (None/int/iterable) into None or a list of ints
def _normalize_axis(axis: Any) -> list[int] | None:
    if axis is None:
        return None
    return [axis] if isinstance(axis, int) else list(axis)


# Backend implementation that delegates array primitives to the compiled Rust extension
class RustBackend(Backend):
    name = "rust"

    # Wraps the compiled Rust backend, raising if the extension isn't built
    def __init__(self) -> None:
        if _rustcore is None:
            raise BackendError("rust_backend_unavailable") from _import_error
        self._impl = _rustcore.RustBackend()
        self._numpy_fallback = NumPyBackend()

    # Returns True if data is a RustArray
    def is_native(self, data: Any) -> bool:
        return isinstance(data, _rustcore.RustArray)

    # Creates a RustArray from raw data
    def array(self, data: Any, dtype: Any = None) -> Any:
        return _to_rust_array(data)

    # Returns a RustArray filled with zeros
    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._impl.zeros(list(shape))

    # Returns a RustArray filled with ones
    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._impl.ones(list(shape))

    # Returns a RustArray filled with fill_value
    def full(self, shape: tuple[int, ...], fill_value: Any, dtype: Any = None) -> Any:
        return self._impl.full(list(shape), float(fill_value))

    # Returns a zero-filled RustArray matching array's shape
    def zeros_like(self, array: Any) -> Any:
        return self._impl.zeros_like(array)

    # Returns a one-filled RustArray matching array's shape
    def ones_like(self, array: Any) -> Any:
        return self._impl.ones_like(array)

    # Reseeds the Rust backend's random number generator
    def seed(self, value: int) -> None:
        self._impl.seed(value)

    # Returns a RustArray of standard-normal random samples
    def randn(self, shape: tuple[int, ...]) -> Any:
        return self._impl.randn(list(shape))

    # Returns a RustArray of uniform [0, 1) random samples
    def rand(self, shape: tuple[int, ...]) -> Any:
        return self._impl.rand(list(shape))

    # Element-wise addition
    def add(self, a: Any, b: Any) -> Any:
        return self._impl.add(a, b)

    # Element-wise subtraction
    def subtract(self, a: Any, b: Any) -> Any:
        return self._impl.subtract(a, b)

    # Element-wise multiplication
    def multiply(self, a: Any, b: Any) -> Any:
        return self._impl.multiply(a, b)

    # Element-wise division
    def divide(self, a: Any, b: Any) -> Any:
        return self._impl.divide(a, b)

    # Element-wise negation
    def negative(self, a: Any) -> Any:
        return self._impl.negative(a)

    # Element-wise absolute value
    def absolute(self, a: Any) -> Any:
        return self._impl.absolute(a)

    # Element-wise exponentiation
    def power(self, a: Any, exponent: Any) -> Any:
        return self._impl.power(a, exponent)

    # Element-wise square root
    def sqrt(self, a: Any) -> Any:
        return self._impl.sqrt(a)

    # Matrix multiplication
    def matmul(self, a: Any, b: Any) -> Any:
        return self._impl.matmul(a, b)

    # Element-wise natural exponential
    def exp(self, a: Any) -> Any:
        return self._impl.exp(a)

    # Element-wise natural logarithm
    def log(self, a: Any) -> Any:
        return self._impl.log(a)

    # Element-wise hyperbolic tangent
    def tanh(self, a: Any) -> Any:
        return self._impl.tanh(a)

    # Element-wise maximum of two arrays
    def maximum(self, a: Any, b: Any) -> Any:
        return self._impl.maximum(a, b)

    # Element-wise a > b comparison
    def greater(self, a: Any, b: Any) -> Any:
        return self._impl.greater(a, b)

    # Sums over the given axis (or all elements)
    def sum(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return self._impl.sum(a, axis=_normalize_axis(axis), keepdims=keepdims)

    # Averages over the given axis (or all elements)
    def mean(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return self._impl.mean(a, axis=_normalize_axis(axis), keepdims=keepdims)

    # Maximum over the given axis (or all elements)
    def max(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return self._impl.max(a, axis=_normalize_axis(axis), keepdims=keepdims)

    # Returns the array reshaped
    def reshape(self, a: Any, shape: tuple[int, ...]) -> Any:
        return self._impl.reshape(a, list(shape))

    # Returns the array with permuted axes
    def transpose(self, a: Any, axes: tuple[int, ...] | None = None) -> Any:
        return self._impl.transpose(a, axes=list(axes) if axes is not None else None)

    # Broadcasts the array to a new shape
    def broadcast_to(self, a: Any, shape: tuple[int, ...]) -> Any:
        return self._impl.broadcast_to(a, list(shape))

    # Extracts sliding windows for Conv2D, delegated to NumPy via array conversion
    def im2col(
        self,
        a: Any,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> Any:
        np_array = np.array(a.tolist(), dtype=np.float64).reshape(a.shape)
        cols = self._numpy_fallback.im2col(np_array, kernel_size, stride, padding)
        return _to_rust_array(cols)

    # Scatter-adds columns back to (N, C, H, W), delegated to NumPy via array conversion
    def col2im(
        self,
        cols: Any,
        input_shape: tuple[int, int, int, int],
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> Any:
        np_cols = np.array(cols.tolist(), dtype=np.float64).reshape(cols.shape)
        result = self._numpy_fallback.col2im(
            np_cols, input_shape, kernel_size, stride, padding
        )
        return _to_rust_array(result)


register_backend_factory("rust", RustBackend)
