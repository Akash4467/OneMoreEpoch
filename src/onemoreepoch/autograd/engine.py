import warnings
from typing import TYPE_CHECKING, Any

from onemoreepoch import config
from onemoreepoch.autograd.graph import topological_order
from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.exceptions import AutogradError, GradientWarning
from onemoreepoch.messages import get_message

if TYPE_CHECKING:
    from onemoreepoch.core.tensor import Tensor

_EXPLOSION_THRESHOLD = 1e3
_VANISHING_THRESHOLD = 1e-7


# Computes gradients of root with respect to every reachable leaf via reverse-mode traversal
def run_backward(root: "Tensor", grad: "Tensor | None" = None) -> None:
    if not root.requires_grad:
        raise AutogradError("backward_no_grad")

    backend = get_backend()
    if grad is None:
        if root.size != 1:
            raise AutogradError("backward_non_scalar", shape=root.shape)
        seed = backend.ones_like(root.data)
    else:
        seed = grad.data if hasattr(grad, "data") else backend.array(grad)

    grads: dict[int, Any] = {id(root): seed}
    health = {"explosion": False, "vanishing": False}

    for tensor in topological_order(root):
        upstream = grads.pop(id(tensor), None)
        if upstream is None:
            continue

        _accumulate_into(tensor, upstream, backend)

        if tensor.creator is None:
            continue

        input_grads = tensor.creator.backward(tensor.context, upstream)
        for parent, parent_grad in zip(tensor.parents, input_grads):
            if parent_grad is None or not parent.requires_grad:
                continue
            if config.debug_checks_enabled():
                _check_gradient_health(parent_grad, backend, health)
            key = id(parent)
            if key in grads:
                grads[key] = backend.add(grads[key], parent_grad)
            else:
                grads[key] = parent_grad


# Warns once per backward call on exploding or vanishing gradients
def _check_gradient_health(grad: Any, backend: Any, health: dict[str, bool]) -> None:
    peak = float(backend.max(backend.absolute(grad)))
    if not health["explosion"] and peak > _EXPLOSION_THRESHOLD:
        health["explosion"] = True
        warnings.warn(
            get_message("gradient_explosion", value=peak),
            GradientWarning,
            stacklevel=4,
        )
    elif not health["vanishing"] and 0.0 < peak < _VANISHING_THRESHOLD:
        health["vanishing"] = True
        warnings.warn(
            get_message("vanishing_gradient", value=peak),
            GradientWarning,
            stacklevel=4,
        )


# Adds grad into tensor.grad, initializing it if needed
def _accumulate_into(tensor: "Tensor", grad: Any, backend: Any) -> None:
    if tensor.grad is None:
        tensor.grad = grad
    else:
        tensor.grad = backend.add(tensor.grad, grad)
