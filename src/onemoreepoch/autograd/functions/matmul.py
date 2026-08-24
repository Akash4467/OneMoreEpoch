"""z = a @ b (2-D matrices)"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.exceptions import ShapeError


class MatMul(Function):
    """z = a @ b (2-D matrices)"""

    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        # Pre-check inner dims for a clear message ("ye shaadi nahi ho sakti").
        if a.shape[-1] != b.shape[0 if b.ndim == 1 else -2]:
            raise ShapeError("shape_mismatch_matmul", left=a.shape, right=b.shape)
        ctx.save_for_backward(a, b)
        return get_backend().matmul(a, b)

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        a, b = ctx.saved_tensors
        backend = get_backend()
        grad_a = backend.matmul(grad, backend.transpose(b))
        grad_b = backend.matmul(backend.transpose(a), grad)
        return grad_a, grad_b
