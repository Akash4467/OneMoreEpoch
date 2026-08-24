"""Inverted dropout: zero elements with probability p, scale survivors."""

from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.module import Module
from onemoreepoch.core.tensor import Tensor
from onemoreepoch.exceptions import ModuleError


class Dropout(Module):
    """Applies inverted dropout during training; a no-op during eval."""

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

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        backend = get_backend()
        keep_prob = 1.0 - self.p
        # greater() yields a 0/1-valued array (same idiom ReLU's backward
        # already relies on) — scale by 1/keep_prob so eval needs no rescale.
        mask = backend.divide(backend.greater(backend.rand(x.shape), self.p), keep_prob)
        return x * Tensor(mask)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
