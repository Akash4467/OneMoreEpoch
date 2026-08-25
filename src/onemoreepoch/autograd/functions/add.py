from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function, unbroadcast
from onemoreepoch.core.backend.registry import get_backend


# Elementwise addition: z = a + b
class Add(Function):
    # Adds a and b, saving their shapes for the backward unbroadcast
    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.extras["shapes"] = (a.shape, b.shape)
        return get_backend().add(a, b)

    # Returns grad unbroadcast to each input's original shape
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        a_shape, b_shape = ctx.extras["shapes"]
        return unbroadcast(grad, a_shape), unbroadcast(grad, b_shape)
