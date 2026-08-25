from typing import Any

import numpy as np

from onemoreepoch.core.backend.base import Backend


# NumPy-backed implementation of the Backend interface
class NumPyBackend(Backend):
    name = "numpy"

    # Initializes a private NumPy random number generator
    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    # Returns True if data is a numpy ndarray
    def is_native(self, data: Any) -> bool:
        return isinstance(data, np.ndarray)

    # Creates a numpy array from raw data
    def array(self, data: Any, dtype: Any = None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    # Returns a numpy array filled with zeros
    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    # Returns a numpy array filled with ones
    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    # Returns a numpy array filled with fill_value
    def full(
        self, shape: tuple[int, ...], fill_value: Any, dtype: Any = None
    ) -> np.ndarray:
        return np.full(shape, fill_value, dtype=dtype)

    # Returns a zero-filled array matching array's shape and dtype
    def zeros_like(self, array: Any) -> np.ndarray:
        return np.zeros_like(array)

    # Returns a one-filled array matching array's shape and dtype
    def ones_like(self, array: Any) -> np.ndarray:
        return np.ones_like(array)

    # Reseeds the random number generator
    def seed(self, value: int) -> None:
        self._rng = np.random.default_rng(value)

    # Returns an array of standard-normal random samples
    def randn(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.standard_normal(shape)

    # Returns an array of uniform [0, 1) random samples
    def rand(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.random(shape)

    # Element-wise addition
    def add(self, a: Any, b: Any) -> np.ndarray:
        return np.add(a, b)

    # Element-wise subtraction
    def subtract(self, a: Any, b: Any) -> np.ndarray:
        return np.subtract(a, b)

    # Element-wise multiplication
    def multiply(self, a: Any, b: Any) -> np.ndarray:
        return np.multiply(a, b)

    # Element-wise division
    def divide(self, a: Any, b: Any) -> np.ndarray:
        return np.divide(a, b)

    # Element-wise negation
    def negative(self, a: Any) -> np.ndarray:
        return np.negative(a)

    # Element-wise absolute value
    def absolute(self, a: Any) -> np.ndarray:
        return np.absolute(a)

    # Element-wise exponentiation
    def power(self, a: Any, exponent: Any) -> np.ndarray:
        return np.power(a, exponent)

    # Element-wise square root
    def sqrt(self, a: Any) -> np.ndarray:
        return np.sqrt(a)

    # Matrix multiplication
    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return np.matmul(a, b)

    # Element-wise natural exponential
    def exp(self, a: Any) -> np.ndarray:
        return np.exp(a)

    # Element-wise natural logarithm
    def log(self, a: Any) -> np.ndarray:
        return np.log(a)

    # Element-wise hyperbolic tangent
    def tanh(self, a: Any) -> np.ndarray:
        return np.tanh(a)

    # Element-wise maximum of two arrays
    def maximum(self, a: Any, b: Any) -> np.ndarray:
        return np.maximum(a, b)

    # Element-wise a > b comparison
    def greater(self, a: Any, b: Any) -> np.ndarray:
        return np.greater(a, b)

    # Sums over the given axis (or all elements)
    def sum(self, a: Any, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.sum(a, axis=axis, keepdims=keepdims)

    # Averages over the given axis (or all elements)
    def mean(self, a: Any, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.mean(a, axis=axis, keepdims=keepdims)

    # Maximum over the given axis (or all elements)
    def max(self, a: Any, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.max(a, axis=axis, keepdims=keepdims)

    # Returns the array reshaped
    def reshape(self, a: Any, shape: tuple[int, ...]) -> np.ndarray:
        return np.reshape(a, shape)

    # Returns the array with permuted axes
    def transpose(self, a: Any, axes: tuple[int, ...] | None = None) -> np.ndarray:
        return np.transpose(a, axes)

    # Broadcasts the array to a new shape
    def broadcast_to(self, a: Any, shape: tuple[int, ...]) -> np.ndarray:
        return np.broadcast_to(a, shape)

    # Extracts sliding windows from a zero-padded (N, C, H, W) array for Conv2D
    def im2col(
        self,
        a: Any,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> np.ndarray:
        kh, kw = kernel_size
        sh, sw = stride
        ph, pw = padding
        padded = np.pad(a, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        n, c, h, w = padded.shape
        h_out = (h - kh) // sh + 1
        w_out = (w - kw) // sw + 1
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (kh, kw), axis=(2, 3)
        )
        windows = windows[:, :, ::sh, ::sw, :, :]
        cols = windows.transpose(0, 1, 4, 5, 2, 3).reshape(
            n, c * kh * kw, h_out * w_out
        )
        return np.ascontiguousarray(cols)

    # Scatter-adds columns back to (N, C, H, W), the adjoint of im2col
    def col2im(
        self,
        cols: Any,
        input_shape: tuple[int, int, int, int],
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> np.ndarray:
        kh, kw = kernel_size
        sh, sw = stride
        ph, pw = padding
        n, c, h, w = input_shape
        h_padded, w_padded = h + 2 * ph, w + 2 * pw
        h_out = (h_padded - kh) // sh + 1
        w_out = (w_padded - kw) // sw + 1
        cols_reshaped = cols.reshape(n, c, kh, kw, h_out, w_out)
        padded = np.zeros((n, c, h_padded, w_padded), dtype=cols.dtype)
        for i in range(kh):
            for j in range(kw):
                padded[
                    :, :, i : i + sh * h_out : sh, j : j + sw * w_out : sw
                ] += cols_reshaped[:, :, i, j, :, :]
        return padded[:, :, ph : ph + h, pw : pw + w]
