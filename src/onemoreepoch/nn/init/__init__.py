from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.tensor import Tensor


# Fills tensor with zeros in place
def zeros_(tensor: Tensor) -> Tensor:
    tensor.data = get_backend().zeros(tensor.shape)
    return tensor


# Fills tensor with ones in place
def ones_(tensor: Tensor) -> Tensor:
    tensor.data = get_backend().ones(tensor.shape)
    return tensor


# Fills tensor in place with samples from Uniform[low, high)
def uniform_(tensor: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    backend = get_backend()
    scaled = backend.multiply(backend.rand(tensor.shape), high - low)
    tensor.data = backend.add(scaled, low)
    return tensor


# Fills tensor in place with samples from Normal(mean, std)
def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    backend = get_backend()
    scaled = backend.multiply(backend.randn(tensor.shape), std)
    tensor.data = backend.add(scaled, mean)
    return tensor


# Fills tensor in place with Uniform[-bound, bound], bound = fan_in ** -0.5
def kaiming_uniform_(tensor: Tensor, fan_in: int | None = None) -> Tensor:
    fan_in = tensor.shape[0] if fan_in is None else fan_in
    bound = fan_in**-0.5
    return uniform_(tensor, -bound, bound)


# Fills tensor in place with Uniform[-bound, bound], bound = sqrt(6 / (fan_in + fan_out))
def xavier_uniform_(
    tensor: Tensor, fan_in: int | None = None, fan_out: int | None = None
) -> Tensor:
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
