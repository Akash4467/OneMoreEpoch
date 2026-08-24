"""z = a ** exponent (exponent is a constant, not a Tensor)"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


class Pow(Function):
    """z = a ** exponent (exponent is a constant, not a Tensor)"""

    @staticmethod
    def forward(ctx: Context, a: Any, *, exponent: float) -> Any:
        ctx.save_for_backward(a)
        ctx.extras["exponent"] = exponent
        return get_backend().power(a, exponent)

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (a,) = ctx.saved_tensors
        exponent = ctx.extras["exponent"]
        backend = get_backend()
        # d(a^n)/da = n * a^(n-1)
        local = backend.multiply(exponent, backend.power(a, exponent - 1))
        return (backend.multiply(grad, local),)
