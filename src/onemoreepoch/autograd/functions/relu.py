from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


# Rectified linear unit: z = max(a, 0)
class ReLU(Function):
    # Clamps a to be non-negative, saving the input for backward
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        out = get_backend().maximum(a, 0)
        ctx.save_for_backward(a)
        return out

    # Zeroes grad wherever the input was non-positive
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        (a,) = ctx.saved_tensors
        backend = get_backend()
        return (backend.multiply(grad, backend.greater(a, 0)),)
