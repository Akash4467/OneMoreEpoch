from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.module import Module
from onemoreepoch.core.tensor import Tensor
from onemoreepoch.exceptions import ModuleError


# Inverted dropout: zeroes elements with probability p and scales survivors during training
class Dropout(Module):
    # Validates p and stores it
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ModuleError(
                "module_param_invalid",
                module="Dropout",
                param="p",
                value=p,
                constraint="0.0 <= p < 1.0",
            )
        self.p = p

    # Applies a random keep-mask during training, or returns x unchanged during eval
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        backend = get_backend()
        keep_prob = 1.0 - self.p
        mask = backend.divide(backend.greater(backend.rand(x.shape), self.p), keep_prob)
        return x * Tensor(mask)

    # Returns a debug string representation
    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
