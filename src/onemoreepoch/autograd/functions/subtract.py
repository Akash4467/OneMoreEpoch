"""z = a - b"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function, unbroadcast
from onemoreepoch.core.backend.registry import get_backend


class Sub(Function):
    """z = a - b"""

    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.extras["shapes"] = (a.shape, b.shape)
        return get_backend().subtract(a, b)

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        a_shape, b_shape = ctx.extras["shapes"]
        backend = get_backend()
        return unbroadcast(grad, a_shape), unbroadcast(backend.negative(grad), b_shape)
