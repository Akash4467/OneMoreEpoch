from typing import Any

import numpy as np

from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.device import Device


# N-dimensional array with optional autograd support
class Tensor:
    # Wraps raw data as a backend array and initializes autograd bookkeeping
    def __init__(
        self,
        data: Any,
        *,
        dtype: Any = None,
        device: Device | None = None,
        requires_grad: bool = False,
    ) -> None:
        backend = get_backend()
        self.data = (
            data
            if dtype is None and backend.is_native(data)
            else backend.array(data, dtype=dtype)
        )
        self.device = device or Device.cpu()
        self.requires_grad = requires_grad
        self.grad: Any = None
        self.creator: Any = None
        self.parents: tuple[Tensor, ...] = ()
        self.context: Any = None

    # Returns a tensor filled with zeros
    @classmethod
    def zeros(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        return cls(get_backend().zeros(shape), requires_grad=requires_grad)

    # Returns a tensor filled with ones
    @classmethod
    def ones(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        return cls(get_backend().ones(shape), requires_grad=requires_grad)

    # Returns a tensor of standard-normal random samples
    @classmethod
    def randn(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        return cls(get_backend().randn(shape), requires_grad=requires_grad)

    # Returns a tensor of uniform [0, 1) random samples
    @classmethod
    def rand(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        return cls(get_backend().rand(shape), requires_grad=requires_grad)

    # Returns the tensor's shape as a tuple
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    # Returns the number of dimensions
    @property
    def ndim(self) -> int:
        return self.data.ndim

    # Returns the underlying dtype
    @property
    def dtype(self) -> Any:
        return self.data.dtype

    # Returns the total number of elements
    @property
    def size(self) -> int:
        return self.data.size

    # Returns the data as an actual NumPy array regardless of backend
    def numpy(self) -> np.ndarray:
        if isinstance(self.data, np.ndarray):
            return self.data
        return np.array(self.data.tolist()).reshape(self.data.shape)

    # Returns the value of a single-element tensor as a Python scalar
    def item(self) -> Any:
        return self.data.item()

    # Returns a new tensor sharing data but detached from the autograd graph
    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    # Computes gradients of this tensor with respect to graph leaves
    def backward(self, grad: "Tensor | None" = None) -> None:
        from onemoreepoch.autograd.engine import run_backward

        run_backward(self, grad)

    # Resets the accumulated gradient to None
    def zero_grad(self) -> None:
        self.grad = None

    # Adds this tensor to another, tracked by autograd
    def __add__(self, other: Any) -> "Tensor":
        from onemoreepoch.autograd.functions import Add

        return Add.apply(self, _ensure_tensor(other))

    # Adds this tensor to another when this tensor is on the right-hand side
    def __radd__(self, other: Any) -> "Tensor":
        return _ensure_tensor(other) + self

    # Subtracts another tensor from this one, tracked by autograd
    def __sub__(self, other: Any) -> "Tensor":
        from onemoreepoch.autograd.functions import Sub

        return Sub.apply(self, _ensure_tensor(other))

    # Subtracts this tensor from another when this tensor is on the right-hand side
    def __rsub__(self, other: Any) -> "Tensor":
        return _ensure_tensor(other) - self

    # Multiplies this tensor by another, tracked by autograd
    def __mul__(self, other: Any) -> "Tensor":
        from onemoreepoch.autograd.functions import Mul

        return Mul.apply(self, _ensure_tensor(other))

    # Multiplies this tensor by another when this tensor is on the right-hand side
    def __rmul__(self, other: Any) -> "Tensor":
        return _ensure_tensor(other) * self

    # Divides this tensor by another, tracked by autograd
    def __truediv__(self, other: Any) -> "Tensor":
        from onemoreepoch.autograd.functions import Div

        return Div.apply(self, _ensure_tensor(other))

    # Divides another tensor by this one when this tensor is on the right-hand side
    def __rtruediv__(self, other: Any) -> "Tensor":
        return _ensure_tensor(other) / self

    # Matrix-multiplies this tensor with another, tracked by autograd
    def __matmul__(self, other: Any) -> "Tensor":
        from onemoreepoch.autograd.functions import MatMul

        return MatMul.apply(self, _ensure_tensor(other))

    # Raises this tensor to a constant power, tracked by autograd
    def __pow__(self, exponent: float) -> "Tensor":
        from onemoreepoch.autograd.functions import Pow

        return Pow.apply(self, exponent=exponent)

    # Negates this tensor, tracked by autograd
    def __neg__(self) -> "Tensor":
        from onemoreepoch.autograd.functions import Neg

        return Neg.apply(self)

    # Computes the elementwise exponential, tracked by autograd
    def exp(self) -> "Tensor":
        from onemoreepoch.autograd.functions import Exp

        return Exp.apply(self)

    # Computes the elementwise natural log, tracked by autograd
    def log(self) -> "Tensor":
        from onemoreepoch.autograd.functions import Log

        return Log.apply(self)

    # Sums elements along an axis (or all axes), tracked by autograd
    def sum(self, axis: Any = None, keepdims: bool = False) -> "Tensor":
        from onemoreepoch.autograd.functions import Sum

        return Sum.apply(self, axis=axis, keepdims=keepdims)

    # Averages elements along an axis (or all axes), tracked by autograd
    def mean(self, axis: Any = None, keepdims: bool = False) -> "Tensor":
        from onemoreepoch.autograd.functions import Mean

        return Mean.apply(self, axis=axis, keepdims=keepdims)

    # Reshapes the tensor, tracked by autograd
    def reshape(self, *shape: int) -> "Tensor":
        from onemoreepoch.autograd.functions import Reshape

        return Reshape.apply(self, shape=shape)

    # Permutes the tensor's axes, tracked by autograd
    def transpose(self, *axes: int) -> "Tensor":
        from onemoreepoch.autograd.functions import Transpose

        return Transpose.apply(self, axes=axes or None)

    # Returns the fully-reversed-axis transpose of the tensor
    @property
    def T(self) -> "Tensor":
        return self.transpose()

    # Returns a debug string representation
    def __repr__(self) -> str:
        grad_info = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data!r}{grad_info})"


# Wraps a raw scalar/array as a Tensor if it isn't one already
def _ensure_tensor(value: Any) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)
