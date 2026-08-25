from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function, unbroadcast
from onemoreepoch.core.backend.registry import get_backend


# Elementwise multiplication: z = a * b
class Mul(Function):
    # Multiplies a and b, saving both for the backward product rule
    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.save_for_backward(a, b)
        return get_backend().multiply(a, b)

    # Returns grad*b and grad*a, each unbroadcast to its input's shape
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        a, b = ctx.saved_tensors
        backend = get_backend()
        grad_a = unbroadcast(backend.multiply(grad, b), a.shape)
        grad_b = unbroadcast(backend.multiply(grad, a), b.shape)
        return grad_a, grad_b
