from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


# Reshapes an array: z = a.reshape(shape)
class Reshape(Function):
    # Reshapes a to shape, saving the original shape for backward
    @staticmethod
    def forward(ctx: Context, a: Any, *, shape: tuple[int, ...]) -> Any:
        ctx.extras["shape"] = a.shape
        return get_backend().reshape(a, shape)

    # Reshapes grad back to the input's original shape
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        return (get_backend().reshape(grad, ctx.extras["shape"]),)
