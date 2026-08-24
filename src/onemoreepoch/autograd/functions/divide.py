"""z = a / b"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function, unbroadcast
from onemoreepoch.core.backend.registry import get_backend


class Div(Function):
    """z = a / b"""

    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.save_for_backward(a, b)
        return get_backend().divide(a, b)

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        a, b = ctx.saved_tensors
        backend = get_backend()
        grad_a = unbroadcast(backend.divide(grad, b), a.shape)
        # d(a/b)/db = -a / b^2
        grad_b = backend.negative(
            backend.divide(backend.multiply(grad, a), backend.multiply(b, b))
        )
        return grad_a, unbroadcast(grad_b, b.shape)
