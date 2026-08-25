from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


# Permutes an array's axes: z = a.transpose(axes)
class Transpose(Function):
    # Transposes a by axes, saving axes for backward
    @staticmethod
    def forward(ctx: Context, a: Any, *, axes: tuple[int, ...] | None = None) -> Any:
        ctx.extras["axes"] = axes
        return get_backend().transpose(a, axes)

    # Transposes grad by the inverse permutation
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        axes = ctx.extras["axes"]
        if axes is None:
            inverse = None
        else:
            inverse = tuple(sorted(range(len(axes)), key=axes.__getitem__))
        return (get_backend().transpose(grad, inverse),)
