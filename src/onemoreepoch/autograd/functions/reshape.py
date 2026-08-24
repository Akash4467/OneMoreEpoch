"""z = a.reshape(shape)"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


class Reshape(Function):
    """z = a.reshape(shape)"""

    @staticmethod
    def forward(ctx: Context, a: Any, *, shape: tuple[int, ...]) -> Any:
        ctx.extras["shape"] = a.shape
        return get_backend().reshape(a, shape)

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        return (get_backend().reshape(grad, ctx.extras["shape"]),)
