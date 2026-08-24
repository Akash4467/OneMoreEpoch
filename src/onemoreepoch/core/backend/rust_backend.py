"""RustBackend: adapts the compiled ``onemoreepoch._rustcore`` extension to
the ``Backend`` ABC.

Import-guarded on purpose: ``import onemoreepoch`` must never fail just
because the Rust extension isn't built. This module is only imported
lazily, when something actually asks the registry for the "rust"
backend (see ``core.backend.registry.get_backend``'s
``core.backend.<name>_backend`` convention) — importing it is what
triggers ``register_backend_factory`` below as a side effect.
"""

from typing import Any

import numpy as np

from onemoreepoch.core.backend.base import Backend
from onemoreepoch.core.backend.factory import register_backend_factory
from onemoreepoch.core.backend.numpy_backend import NumPyBackend
from onemoreepoch.exceptions import BackendError

try:
    from onemoreepoch import _rustcore
except ImportError as exc:  # pragma: no cover - depends on whether it's built
    _rustcore = None
    _import_error = exc
else:
    _import_error = None


def _to_rust_array(data: Any) -> Any:
    """Normalize arbitrary input (list/tuple/scalar/np.ndarray/RustArray) to a RustArray.

    Goes through NumPy purely as a marshalling convenience (nested-list
    parsing, dtype coercion) — not a computational shortcut; every actual
    math operation still runs in Rust.
    """
    if _rustcore is not None and isinstance(data, _rustcore.RustArray):
        return data
    array = np.asarray(data, dtype=np.float64)
    return _rustcore.RustArray.from_flat(array.ravel().tolist(), list(array.shape))


def _normalize_axis(axis: Any) -> list[int] | None:
    if axis is None:
        return None
    return [axis] if isinstance(axis, int) else list(axis)


class RustBackend(Backend):
    """Delegates array primitives to the compiled Rust extension.

    ``im2col``/``col2im`` (Conv2D's windowed-extraction primitives) are the
    one deliberate exception: they convert through NumPy and delegate to
    ``NumPyBackend`` rather than a hand-rolled Rust implementation — see
    ``rust/onemoreepoch-core/src/backend.rs``'s module docstring for why.
    """

    name = "rust"

    def __init__(self) -> None:
        if _rustcore is None:
            raise BackendError("rust_backend_unavailable") from _import_error
        self._impl = _rustcore.RustBackend()
        self._numpy_fallback = NumPyBackend()

    # -- array creation ---------------------------------------------------

    def is_native(self, data: Any) -> bool:
        return isinstance(data, _rustcore.RustArray)

    def array(self, data: Any, dtype: Any = None) -> Any:
        return _to_rust_array(data)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._impl.zeros(list(shape))

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._impl.ones(list(shape))

    def full(self, shape: tuple[int, ...], fill_value: Any, dtype: Any = None) -> Any:
        return self._impl.full(list(shape), float(fill_value))

    def zeros_like(self, array: Any) -> Any:
        return self._impl.zeros_like(array)

    def ones_like(self, array: Any) -> Any:
        return self._impl.ones_like(array)

    # -- random ------------------------------------------------------------

    def seed(self, value: int) -> None:
        self._impl.seed(value)

    def randn(self, shape: tuple[int, ...]) -> Any:
        return self._impl.randn(list(shape))

    def rand(self, shape: tuple[int, ...]) -> Any:
        return self._impl.rand(list(shape))

    # -- arithmetic ----------------------------------------------------------

    def add(self, a: Any, b: Any) -> Any:
        return self._impl.add(a, b)

    def subtract(self, a: Any, b: Any) -> Any:
        return self._impl.subtract(a, b)

    def multiply(self, a: Any, b: Any) -> Any:
        return self._impl.multiply(a, b)

    def divide(self, a: Any, b: Any) -> Any:
        return self._impl.divide(a, b)

    def negative(self, a: Any) -> Any:
        return self._impl.negative(a)

    def absolute(self, a: Any) -> Any:
        return self._impl.absolute(a)

    def power(self, a: Any, exponent: Any) -> Any:
        return self._impl.power(a, exponent)

    def sqrt(self, a: Any) -> Any:
        return self._impl.sqrt(a)

    def matmul(self, a: Any, b: Any) -> Any:
        return self._impl.matmul(a, b)

    def exp(self, a: Any) -> Any:
        return self._impl.exp(a)

    def log(self, a: Any) -> Any:
        return self._impl.log(a)

    def tanh(self, a: Any) -> Any:
        return self._impl.tanh(a)

    def maximum(self, a: Any, b: Any) -> Any:
        return self._impl.maximum(a, b)

    def greater(self, a: Any, b: Any) -> Any:
        return self._impl.greater(a, b)

    # -- reductions ------------------------------------------------------

    def sum(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return self._impl.sum(a, axis=_normalize_axis(axis), keepdims=keepdims)

    def mean(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return self._impl.mean(a, axis=_normalize_axis(axis), keepdims=keepdims)

    def max(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return self._impl.max(a, axis=_normalize_axis(axis), keepdims=keepdims)

    # -- shape manipulation ------------------------------------------------

    def reshape(self, a: Any, shape: tuple[int, ...]) -> Any:
        return self._impl.reshape(a, list(shape))

    def transpose(self, a: Any, axes: tuple[int, ...] | None = None) -> Any:
        return self._impl.transpose(a, axes=list(axes) if axes is not None else None)

    def broadcast_to(self, a: Any, shape: tuple[int, ...]) -> Any:
        return self._impl.broadcast_to(a, list(shape))

    # -- windowed extraction (Conv2D) -- NumPy-delegated, see class docstring

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


# Registered unconditionally — RustBackend() itself raises the specific,
# actionable BackendError("rust_backend_unavailable") when _rustcore is
# None, which is more helpful than falling through to the registry's
# generic "unknown_backend" error.
register_backend_factory("rust", RustBackend)
