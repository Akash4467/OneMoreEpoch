"""z = tanh(a)"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


class Tanh(Function):
    """z = tanh(a)"""

    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        out = get_backend().tanh(a)
        ctx.save_for_backward(out)  # derivative reuses the output: 1 - t^2
        return out

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (out,) = ctx.saved_tensors
        backend = get_backend()
        local = backend.subtract(1, backend.multiply(out, out))
        return (backend.multiply(grad, local),)
