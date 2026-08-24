"""Loss functions."""

from onemoreepoch.core.tensor import Tensor
from onemoreepoch.core.module import Module


class MSELoss(Module):
    """Mean squared error: mean((prediction - target)^2)."""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = prediction - target
        return (diff * diff).mean()

    def __repr__(self) -> str:
        return "MSELoss()"


__all__ = ["MSELoss"]
