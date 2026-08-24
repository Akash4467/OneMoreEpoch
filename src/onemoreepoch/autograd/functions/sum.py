"""z = sum(a)"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.autograd.functions._reduction import expand_reduced_grad
from onemoreepoch.core.backend.registry import get_backend


class Sum(Function):
    """z = sum(a)"""

    @staticmethod
    def forward(
        ctx: Context, a: Any, *, axis: Any = None, keepdims: bool = False
    ) -> Any:
        ctx.extras.update(shape=a.shape, axis=axis, keepdims=keepdims)
        return get_backend().sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        e = ctx.extras
        return (expand_reduced_grad(grad, e["shape"], e["axis"], e["keepdims"]),)
