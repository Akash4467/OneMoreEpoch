from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function, unbroadcast
from onemoreepoch.core.backend.registry import get_backend


# Elementwise subtraction: z = a - b
class Sub(Function):
    # Subtracts b from a, saving their shapes for the backward unbroadcast
    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.extras["shapes"] = (a.shape, b.shape)
        return get_backend().subtract(a, b)

    # Returns grad unbroadcast to a's shape and -grad unbroadcast to b's shape
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        a_shape, b_shape = ctx.extras["shapes"]
        backend = get_backend()
        return unbroadcast(grad, a_shape), unbroadcast(backend.negative(grad), b_shape)
