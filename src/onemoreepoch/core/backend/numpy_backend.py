"""NumPy execution backend."""

from typing import Any

import numpy as np

from onemoreepoch.core.backend.base import Backend


class NumPyBackend(Backend):
    """Backend implementation backed by NumPy."""

    name = "numpy"

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    # -- array creation -------------------------------------------------

    def is_native(self, data: Any) -> bool:
        return isinstance(data, np.ndarray)

    def array(self, data: Any, dtype: Any = None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def full(
        self, shape: tuple[int, ...], fill_value: Any, dtype: Any = None
    ) -> np.ndarray:
        return np.full(shape, fill_value, dtype=dtype)

    def zeros_like(self, array: Any) -> np.ndarray:
        return np.zeros_like(array)

    def ones_like(self, array: Any) -> np.ndarray:
        return np.ones_like(array)

    # -- random ----------------------------------------------------------

    def seed(self, value: int) -> None:
        self._rng = np.random.default_rng(value)

    def randn(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.standard_normal(shape)

    def rand(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.random(shape)

    # -- arithmetic ------------------------------------------------------

    def add(self, a: Any, b: Any) -> np.ndarray:
        return np.add(a, b)

    def subtract(self, a: Any, b: Any) -> np.ndarray:
        return np.subtract(a, b)

    def multiply(self, a: Any, b: Any) -> np.ndarray:
        return np.multiply(a, b)

    def divide(self, a: Any, b: Any) -> np.ndarray:
        return np.divide(a, b)

    def negative(self, a: Any) -> np.ndarray:
        return np.negative(a)

    def absolute(self, a: Any) -> np.ndarray:
        return np.absolute(a)

    def power(self, a: Any, exponent: Any) -> np.ndarray:
        return np.power(a, exponent)

    def sqrt(self, a: Any) -> np.ndarray:
        return np.sqrt(a)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return np.matmul(a, b)

    def exp(self, a: Any) -> np.ndarray:
        return np.exp(a)

    def log(self, a: Any) -> np.ndarray:
        return np.log(a)

    def tanh(self, a: Any) -> np.ndarray:
        return np.tanh(a)

    def maximum(self, a: Any, b: Any) -> np.ndarray:
        return np.maximum(a, b)

    def greater(self, a: Any, b: Any) -> np.ndarray:
        return np.greater(a, b)

    # -- reductions ------------------------------------------------------

    def sum(self, a: Any, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a: Any, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.mean(a, axis=axis, keepdims=keepdims)

    def max(self, a: Any, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.max(a, axis=axis, keepdims=keepdims)

    # -- shape manipulation ----------------------------------------------

    def reshape(self, a: Any, shape: tuple[int, ...]) -> np.ndarray:
        return np.reshape(a, shape)

    def transpose(self, a: Any, axes: tuple[int, ...] | None = None) -> np.ndarray:
        return np.transpose(a, axes)

    def broadcast_to(self, a: Any, shape: tuple[int, ...]) -> np.ndarray:
        return np.broadcast_to(a, shape)

    # -- windowed extraction (Conv2D) -------------------------------------

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
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))
        windows = windows[:, :, ::sh, ::sw, :, :]
        cols = windows.transpose(0, 1, 4, 5, 2, 3).reshape(n, c * kh * kw, h_out * w_out)
        return np.ascontiguousarray(cols)

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
                padded[:, :, i : i + sh * h_out : sh, j : j + sw * w_out : sw] += (
                    cols_reshaped[:, :, i, j, :, :]
                )
        return padded[:, :, ph : ph + h, pw : pw + w]
