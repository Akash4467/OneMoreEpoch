from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


# Raises to a constant power: z = a ** exponent
class Pow(Function):
    # Raises a to exponent, saving both for the backward power rule
    @staticmethod
    def forward(ctx: Context, a: Any, *, exponent: float) -> Any:
        ctx.save_for_backward(a)
        ctx.extras["exponent"] = exponent
        return get_backend().power(a, exponent)

    # Returns grad * exponent * a**(exponent - 1)
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (a,) = ctx.saved_tensors
        exponent = ctx.extras["exponent"]
        backend = get_backend()
        local = backend.multiply(exponent, backend.power(a, exponent - 1))
        return (backend.multiply(grad, local),)
