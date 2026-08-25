from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.autograd.functions._reduction import expand_reduced_grad
from onemoreepoch.core.backend.registry import get_backend


# Reduction: z = mean(a) over an axis (or all axes)
class Mean(Function):
    # Averages a over axis, saving shape/axis/keepdims for backward
    @staticmethod
    def forward(
        ctx: Context, a: Any, *, axis: Any = None, keepdims: bool = False
    ) -> Any:
        ctx.extras.update(shape=a.shape, axis=axis, keepdims=keepdims)
        return get_backend().mean(a, axis=axis, keepdims=keepdims)

    # Divides grad by the reduced element count, then broadcasts back to the input's shape
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        e = ctx.extras
        shape, axis = e["shape"], e["axis"]
        backend = get_backend()
        if axis is None:
            count = 1
            for dim in shape:
                count *= dim
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = 1
            for ax in axes:
                count *= shape[ax]
        grad = backend.divide(grad, count)
        return (expand_reduced_grad(grad, shape, axis, e["keepdims"]),)
