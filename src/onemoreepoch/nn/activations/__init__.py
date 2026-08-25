from onemoreepoch.autograd import functions
from onemoreepoch.core.module import Module
from onemoreepoch.core.tensor import Tensor


# Applies the rectified linear unit element-wise
class ReLU(Module):
    # Runs the ReLU autograd function on x
    def forward(self, x: Tensor) -> Tensor:
        return functions.ReLU.apply(x)

    # Returns a debug string representation
    def __repr__(self) -> str:
        return "ReLU()"


# Applies the logistic sigmoid element-wise
class Sigmoid(Module):
    # Runs the Sigmoid autograd function on x
    def forward(self, x: Tensor) -> Tensor:
        return functions.Sigmoid.apply(x)

    # Returns a debug string representation
    def __repr__(self) -> str:
        return "Sigmoid()"


# Applies the hyperbolic tangent element-wise
class Tanh(Module):
    # Runs the Tanh autograd function on x
    def forward(self, x: Tensor) -> Tensor:
        return functions.Tanh.apply(x)

    # Returns a debug string representation
    def __repr__(self) -> str:
        return "Tanh()"


__all__ = ["ReLU", "Sigmoid", "Tanh"]
