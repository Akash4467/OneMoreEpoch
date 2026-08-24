"""z = max(a, 0)"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


class ReLU(Function):
    """z = max(a, 0)"""

    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        out = get_backend().maximum(a, 0)
        ctx.save_for_backward(a)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (a,) = ctx.saved_tensors
        backend = get_backend()
        return (backend.multiply(grad, backend.greater(a, 0)),)
