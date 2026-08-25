from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


# Elementwise negation: z = -a
class Neg(Function):
    # Negates a
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        return get_backend().negative(a)

    # Returns the negated upstream gradient
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        return (get_backend().negative(grad),)
