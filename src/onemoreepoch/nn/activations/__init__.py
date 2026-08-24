"""Activation modules — thin wrappers over autograd functions."""

from onemoreepoch.autograd import functions
from onemoreepoch.core.module import Module
from onemoreepoch.core.tensor import Tensor


class ReLU(Module):
    """Applies the rectified linear unit element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        return functions.ReLU.apply(x)

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Applies the logistic sigmoid element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        return functions.Sigmoid.apply(x)

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """Applies the hyperbolic tangent element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        return functions.Tanh.apply(x)

    def __repr__(self) -> str:
        return "Tanh()"


__all__ = ["ReLU", "Sigmoid", "Tanh"]
