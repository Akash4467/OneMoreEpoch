"""z = e^a"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


class Exp(Function):
    """z = e^a"""

    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        out = get_backend().exp(a)
        ctx.save_for_backward(out)  # d(e^a)/da = e^a — save the output itself
        return out

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (out,) = ctx.saved_tensors
        return (get_backend().multiply(grad, out),)
