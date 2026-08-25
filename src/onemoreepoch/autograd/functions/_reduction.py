from typing import Any

from onemoreepoch.core.backend.registry import get_backend


# Broadcasts a reduced gradient back to the input's original shape
def expand_reduced_grad(
    grad: Any, input_shape: tuple[int, ...], axis: Any, keepdims: bool
) -> Any:
    backend = get_backend()
    if axis is not None and not keepdims:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        shape = list(input_shape)
        for ax in sorted(ax % len(input_shape) for ax in axes):
            shape[ax] = 1
        grad = backend.reshape(grad, tuple(shape))
    return backend.broadcast_to(grad, input_shape)
