from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


# Logistic sigmoid: z = 1 / (1 + e^-a)
class Sigmoid(Function):
    # Computes the sigmoid of a, saving the output since the derivative reuses it
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        backend = get_backend()
        out = backend.divide(1, backend.add(1, backend.exp(backend.negative(a))))
        ctx.save_for_backward(out)
        return out

    # Returns grad * output * (1 - output)
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (out,) = ctx.saved_tensors
        backend = get_backend()
        local = backend.multiply(out, backend.subtract(1, out))
        return (backend.multiply(grad, local),)
