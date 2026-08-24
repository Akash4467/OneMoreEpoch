"""z = -a"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


class Neg(Function):
    """z = -a"""

    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        return get_backend().negative(a)

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        return (get_backend().negative(grad),)
