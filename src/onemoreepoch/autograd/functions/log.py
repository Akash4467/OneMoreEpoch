"""z = ln(a)"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


class Log(Function):
    """z = ln(a)"""

    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        ctx.save_for_backward(a)
        return get_backend().log(a)

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (a,) = ctx.saved_tensors
        return (get_backend().divide(grad, a),)
