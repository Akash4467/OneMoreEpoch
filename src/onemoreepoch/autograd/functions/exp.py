from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


# Elementwise natural exponential: z = e^a
class Exp(Function):
    # Computes e^a, saving the output since its own derivative is itself
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        out = get_backend().exp(a)
        ctx.save_for_backward(out)
        return out

    # Returns grad * output
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (out,) = ctx.saved_tensors
        return (get_backend().multiply(grad, out),)
