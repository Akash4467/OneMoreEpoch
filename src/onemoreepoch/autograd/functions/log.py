from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


# Elementwise natural logarithm: z = ln(a)
class Log(Function):
    # Computes ln(a), saving the input for the backward quotient
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        ctx.save_for_backward(a)
        return get_backend().log(a)

    # Returns grad / a
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (a,) = ctx.saved_tensors
        return (get_backend().divide(grad, a),)
