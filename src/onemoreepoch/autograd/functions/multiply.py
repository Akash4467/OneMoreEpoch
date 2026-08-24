"""z = a * b"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function, unbroadcast
from onemoreepoch.core.backend.registry import get_backend


class Mul(Function):
    """z = a * b"""

    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.save_for_backward(a, b)
        return get_backend().multiply(a, b)

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        a, b = ctx.saved_tensors
        backend = get_backend()
        grad_a = unbroadcast(backend.multiply(grad, b), a.shape)
        grad_b = unbroadcast(backend.multiply(grad, a), b.shape)
        return grad_a, grad_b
