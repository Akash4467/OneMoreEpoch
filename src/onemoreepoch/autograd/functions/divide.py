from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function, unbroadcast
from onemoreepoch.core.backend.registry import get_backend


# Elementwise division: z = a / b
class Div(Function):
    # Divides a by b, saving both for the backward quotient rule
    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.save_for_backward(a, b)
        return get_backend().divide(a, b)

    # Returns d(a/b)/da and d(a/b)/db, each unbroadcast to its input's shape
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        a, b = ctx.saved_tensors
        backend = get_backend()
        grad_a = unbroadcast(backend.divide(grad, b), a.shape)
        grad_b = backend.negative(
            backend.divide(backend.multiply(grad, a), backend.multiply(b, b))
        )
        return grad_a, unbroadcast(grad_b, b.shape)
