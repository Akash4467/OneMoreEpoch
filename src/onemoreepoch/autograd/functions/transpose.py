"""z = a.transpose(axes)"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


class Transpose(Function):
    """z = a.transpose(axes)"""

    @staticmethod
    def forward(ctx: Context, a: Any, *, axes: tuple[int, ...] | None = None) -> Any:
        ctx.extras["axes"] = axes
        return get_backend().transpose(a, axes)

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        axes = ctx.extras["axes"]
        if axes is None:
            inverse = None  # full reversal is its own inverse
        else:
            inverse = tuple(sorted(range(len(axes)), key=axes.__getitem__))
        return (get_backend().transpose(grad, inverse),)
