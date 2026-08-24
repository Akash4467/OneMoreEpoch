"""Weight initialization schemes.

Each function mutates a Tensor's ``.data`` in place and returns it
(PyTorch's trailing-underscore convention), going entirely through
``Backend`` methods rather than raw Python operators — so these work
identically regardless of which backend produced the tensor's storage.
"""

from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.tensor import Tensor


def zeros_(tensor: Tensor) -> Tensor:
    """Fill ``tensor`` with zeros."""
    tensor.data = get_backend().zeros(tensor.shape)
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    """Fill ``tensor`` with ones."""
    tensor.data = get_backend().ones(tensor.shape)
    return tensor


def uniform_(tensor: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    """Fill ``tensor`` with samples from Uniform[low, high)."""
    backend = get_backend()
    scaled = backend.multiply(backend.rand(tensor.shape), high - low)
    tensor.data = backend.add(scaled, low)
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill ``tensor`` with samples from Normal(mean, std)."""
    backend = get_backend()
    scaled = backend.multiply(backend.randn(tensor.shape), std)
    tensor.data = backend.add(scaled, mean)
    return tensor


def kaiming_uniform_(tensor: Tensor, fan_in: int | None = None) -> Tensor:
    """Uniform[-bound, bound] with bound = fan_in ** -0.5.

    ``fan_in`` defaults to ``tensor.shape[0]`` but should be passed
    explicitly for tensors (e.g. a bias) whose own shape doesn't carry
    the layer's fan-in.
    """
    fan_in = tensor.shape[0] if fan_in is None else fan_in
    bound = fan_in**-0.5
    return uniform_(tensor, -bound, bound)


def xavier_uniform_(
    tensor: Tensor, fan_in: int | None = None, fan_out: int | None = None
) -> Tensor:
    """Uniform[-bound, bound] with bound = sqrt(6 / (fan_in + fan_out))."""
    shape = tensor.shape
    fan_in = shape[0] if fan_in is None else fan_in
    fan_out = (shape[1] if len(shape) > 1 else shape[0]) if fan_out is None else fan_out
    bound = (6.0 / (fan_in + fan_out)) ** 0.5
    return uniform_(tensor, -bound, bound)


__all__ = [
    "kaiming_uniform_",
    "normal_",
    "ones_",
    "uniform_",
    "xavier_uniform_",
    "zeros_",
]
