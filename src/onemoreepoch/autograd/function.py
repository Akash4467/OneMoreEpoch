"""Function: the base class for all differentiable operations.

Every mathematical operation is a stateless Function subclass (ADR-005).
``apply()`` is the single entry point: it runs forward on raw arrays,
wraps the result in a Tensor, and wires up the graph pointers. State
needed by backward lives only in the per-call Context (ADR-006).
"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.tensor import Tensor
from onemoreepoch.exceptions import OneMoreEpochError, ShapeError


class Function:
    """One differentiable operation. Subclasses define forward/backward."""

    @staticmethod
    def forward(ctx: Context, *arrays: Any, **kwargs: Any) -> Any:
        """Compute the output array from input arrays.

        Receives raw backend arrays, never Tensors. Saves anything
        backward will need into ``ctx``.
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        """Return the gradient w.r.t. each input array.

        ``grad`` is the upstream gradient of the output. Must return one
        entry per tensor input (None for non-differentiable inputs).
        """
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: Tensor, **kwargs: Any) -> Tensor:
        """Run forward and wire the output into the computation graph."""
        ctx = Context()
        raw_inputs = tuple(t.data for t in inputs)
        try:
            output_data = cls.forward(ctx, *raw_inputs, **kwargs)
        except OneMoreEpochError:
            raise  # already a friendly, mode-aware message
        except ValueError as exc:
            # Backend broadcasting errors get one central, friendly wrapper.
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


def unbroadcast(grad: Any, shape: tuple[int, ...]) -> Any:
    """Reduce ``grad`` back to ``shape`` by summing broadcasted axes.

    Forward ops rely on backend broadcasting (e.g. (3,4) + (4,)); the
    gradient flowing back has the broadcasted shape and must be summed
    down to each input's original shape.
    """
    backend = get_backend()
    # Sum away leading axes added by broadcasting.
    while grad.ndim > len(shape):
        grad = backend.sum(grad, axis=0)
    # Sum axes that were size-1 in the original shape.
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = backend.sum(grad, axis=axis, keepdims=True)
    return grad
