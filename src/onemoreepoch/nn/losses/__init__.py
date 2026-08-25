from onemoreepoch.core.module import Module
from onemoreepoch.core.tensor import Tensor


# Mean squared error loss: mean((prediction - target)^2)
class MSELoss(Module):
    # Computes the mean squared difference between prediction and target
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = prediction - target
        return (diff * diff).mean()

    # Returns a debug string representation
    def __repr__(self) -> str:
        return "MSELoss()"


__all__ = ["MSELoss"]
