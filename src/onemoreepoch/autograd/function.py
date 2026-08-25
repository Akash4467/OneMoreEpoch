from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.tensor import Tensor
from onemoreepoch.exceptions import OneMoreEpochError, ShapeError


# Base class for one differentiable operation; subclasses define forward/backward
class Function:
    # Computes the output array from raw input arrays, saving state into ctx for backward
    @staticmethod
    def forward(ctx: Context, *arrays: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # Returns the gradient with respect to each input array
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        raise NotImplementedError

    # Runs forward and wires the resulting Tensor into the computation graph
    @classmethod
    def apply(cls, *inputs: Tensor, **kwargs: Any) -> Tensor:
        ctx = Context()
        raw_inputs = tuple(t.data for t in inputs)
        try:
            output_data = cls.forward(ctx, *raw_inputs, **kwargs)
        except OneMoreEpochError:
            raise
        except ValueError as exc:
            raise ShapeError(
                "broadcast_failure",
                op=cls.__name__,
                shapes=tuple(arr.shape for arr in raw_inputs),
            ) from exc

        requires_grad = any(t.requires_grad for t in inputs)
        output = Tensor(output_data, requires_grad=requires_grad)
        if requires_grad:
            output.creator = cls
            output.parents = inputs
            output.context = ctx
        return output


# Sums grad's broadcasted axes back down to match shape
def unbroadcast(grad: Any, shape: tuple[int, ...]) -> Any:
    backend = get_backend()
    while grad.ndim > len(shape):
        grad = backend.sum(grad, axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = backend.sum(grad, axis=axis, keepdims=True)
    return grad
