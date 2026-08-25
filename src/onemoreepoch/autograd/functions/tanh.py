from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


# Hyperbolic tangent: z = tanh(a)
class Tanh(Function):
    # Computes tanh(a), saving the output since the derivative reuses it
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        out = get_backend().tanh(a)
        ctx.save_for_backward(out)
        return out

    # Returns grad * (1 - output^2)
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (out,) = ctx.saved_tensors
        backend = get_backend()
        local = backend.subtract(1, backend.multiply(out, out))
        return (backend.multiply(grad, local),)
