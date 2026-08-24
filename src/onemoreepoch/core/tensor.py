"""Tensor: the core n-dimensional array with autograd bookkeeping.

Tensor stores data and passive graph pointers (``creator``/``parents``/
``context``) but contains no gradient math itself. Operator overloads
dispatch to autograd ``Function`` subclasses, and ``backward()`` delegates
to ``autograd.engine`` (ADR-008 — the one documented upward call).
"""

from typing import Any

import numpy as np

from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.device import Device


class Tensor:
    """N-dimensional array with optional autograd support."""

    def __init__(
        self,
        data: Any,
        *,
        dtype: Any = None,
        device: Device | None = None,
        requires_grad: bool = False,
    ) -> None:
        backend = get_backend()
        # Re-wrapping an existing backend array is a no-op copy avoidance.
        self.data = (
            data
            if dtype is None and backend.is_native(data)
            else backend.array(data, dtype=dtype)
        )
        self.device = device or Device.cpu()
        self.requires_grad = requires_grad
        self.grad: Any = None
        # Graph pointers — written by Function.apply(), read by the engine.
        self.creator: Any = None
        self.parents: tuple[Tensor, ...] = ()
        self.context: Any = None

    # -- factories -------------------------------------------------------

    @classmethod
    def zeros(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        """Return a tensor filled with zeros."""
        return cls(get_backend().zeros(shape), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        """Return a tensor filled with ones."""
        return cls(get_backend().ones(shape), requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        """Return a tensor of standard-normal samples."""
        return cls(get_backend().randn(shape), requires_grad=requires_grad)

    @classmethod
    def rand(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        """Return a tensor of uniform [0, 1) samples."""
        return cls(get_backend().rand(shape), requires_grad=requires_grad)

    # -- metadata --------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    # -- conversion ------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """Return the data as an actual NumPy array, regardless of backend.

        Under NumPyBackend ``.data`` already is one; under any other
        backend (e.g. Rust) it's converted via ``tolist()``/``shape`` —
        this is the one place Tensor is allowed to know about NumPy
        specifically, since that's the entire point of this method.
        """
        if isinstance(self.data, np.ndarray):
            return self.data
        return np.array(self.data.tolist()).reshape(self.data.shape)

    def item(self) -> Any:
        """Return the value of a single-element tensor as a Python scalar."""
        return self.data.item()

    def detach(self) -> "Tensor":
        """Return a new tensor sharing data but cut off from the graph."""
        return Tensor(self.data, requires_grad=False)

    # -- autograd --------------------------------------------------------

    def backward(self, grad: "Tensor | None" = None) -> None:
        """Compute gradients of this tensor w.r.t. graph leaves.

        Thin delegator into the autograd engine (ADR-008).
        """
        from onemoreepoch.autograd.engine import run_backward

        run_backward(self, grad)

    def zero_grad(self) -> None:
        """Reset the accumulated gradient."""
        self.grad = None

    # -- operator overloads (dispatch to autograd Functions) -------------

    def __add__(self, other: Any) -> "Tensor":
        from onemoreepoch.autograd.functions import Add

        return Add.apply(self, _ensure_tensor(other))

    def __radd__(self, other: Any) -> "Tensor":
        return _ensure_tensor(other) + self

    def __sub__(self, other: Any) -> "Tensor":
        from onemoreepoch.autograd.functions import Sub

        return Sub.apply(self, _ensure_tensor(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return _ensure_tensor(other) - self

    def __mul__(self, other: Any) -> "Tensor":
        from onemoreepoch.autograd.functions import Mul

        return Mul.apply(self, _ensure_tensor(other))

    def __rmul__(self, other: Any) -> "Tensor":
        return _ensure_tensor(other) * self

    def __truediv__(self, other: Any) -> "Tensor":
        from onemoreepoch.autograd.functions import Div

        return Div.apply(self, _ensure_tensor(other))

    def __rtruediv__(self, other: Any) -> "Tensor":
        return _ensure_tensor(other) / self

    def __matmul__(self, other: Any) -> "Tensor":
        from onemoreepoch.autograd.functions import MatMul

        return MatMul.apply(self, _ensure_tensor(other))

    def __pow__(self, exponent: float) -> "Tensor":
        from onemoreepoch.autograd.functions import Pow

        return Pow.apply(self, exponent=exponent)

    def __neg__(self) -> "Tensor":
        from onemoreepoch.autograd.functions import Neg

        return Neg.apply(self)

    # -- math methods ----------------------------------------------------

    def exp(self) -> "Tensor":
        from onemoreepoch.autograd.functions import Exp

        return Exp.apply(self)

    def log(self) -> "Tensor":
        from onemoreepoch.autograd.functions import Log

        return Log.apply(self)

    def sum(self, axis: Any = None, keepdims: bool = False) -> "Tensor":
        from onemoreepoch.autograd.functions import Sum

        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Any = None, keepdims: bool = False) -> "Tensor":
        from onemoreepoch.autograd.functions import Mean

        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: int) -> "Tensor":
        from onemoreepoch.autograd.functions import Reshape

        return Reshape.apply(self, shape=shape)

    def transpose(self, *axes: int) -> "Tensor":
        from onemoreepoch.autograd.functions import Transpose

        return Transpose.apply(self, axes=axes or None)

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    # -- repr ------------------------------------------------------------

    def __repr__(self) -> str:
        grad_info = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data!r}{grad_info})"


def _ensure_tensor(value: Any) -> Tensor:
    """Wrap raw scalars/arrays so binary ops always see two Tensors."""
    return value if isinstance(value, Tensor) else Tensor(value)
