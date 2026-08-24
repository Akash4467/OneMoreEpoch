"""The backward engine: reverse topological graph traversal.

Owns all graph-traversal logic (ADR-008). Walks ``creator``/``parents``
pointers back from the root tensor, invoking each Function's backward
and accumulating gradients into ``Tensor.grad``.
"""

import warnings
from typing import TYPE_CHECKING, Any

from onemoreepoch import config
from onemoreepoch.autograd.graph import topological_order
from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.exceptions import AutogradError, GradientWarning
from onemoreepoch.messages import get_message

if TYPE_CHECKING:
    from onemoreepoch.core.tensor import Tensor

# Thresholds for the opt-in gradient-health diagnostics.
_EXPLOSION_THRESHOLD = 1e3
_VANISHING_THRESHOLD = 1e-7


def run_backward(root: "Tensor", grad: "Tensor | None" = None) -> None:
    """Compute gradients of ``root`` w.r.t. every reachable leaf.

    ``grad`` seeds the traversal; it defaults to ones (which requires
    ``root`` to be a scalar, matching PyTorch semantics).
    """
    if not root.requires_grad:
        raise AutogradError("backward_no_grad")

    backend = get_backend()
    if grad is None:
        if root.size != 1:
            raise AutogradError("backward_non_scalar", shape=root.shape)
        seed = backend.ones_like(root.data)
    else:
        seed = grad.data if hasattr(grad, "data") else backend.array(grad)

    # grad accumulation buffer keyed by tensor identity
    grads: dict[int, Any] = {id(root): seed}
    # Warn at most once per backward call, per condition.
    health = {"explosion": False, "vanishing": False}

    for tensor in topological_order(root):
        upstream = grads.pop(id(tensor), None)
        if upstream is None:
            continue

        # Educational choice: every requires_grad tensor keeps its grad
        # (PyTorch only keeps leaves by default) so users can inspect
        # gradients anywhere in the graph.
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


def _check_gradient_health(grad: Any, backend: Any, health: dict[str, bool]) -> None:
    """Warn (once per backward) on exploding or vanishing gradients."""
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


def _accumulate_into(tensor: "Tensor", grad: Any, backend: Any) -> None:
    """Add ``grad`` into ``tensor.grad``, initializing it if needed."""
    if tensor.grad is None:
        tensor.grad = grad
    else:
        tensor.grad = backend.add(tensor.grad, grad)


