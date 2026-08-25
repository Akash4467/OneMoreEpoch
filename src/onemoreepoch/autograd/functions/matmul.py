from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.exceptions import ShapeError


# Matrix multiplication for 2-D matrices: z = a @ b
class MatMul(Function):
    # Matrix-multiplies a and b, raising a friendly error on inner-dim mismatch
    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        if a.shape[-1] != b.shape[0 if b.ndim == 1 else -2]:
            raise ShapeError("shape_mismatch_matmul", left=a.shape, right=b.shape)
        ctx.save_for_backward(a, b)
        return get_backend().matmul(a, b)

    # Returns grad @ b^T and a^T @ grad
    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        a, b = ctx.saved_tensors
        backend = get_backend()
        grad_a = backend.matmul(grad, backend.transpose(b))
        grad_b = backend.matmul(backend.transpose(a), grad)
        return grad_a, grad_b
