"""Optimizer: template-method base class for parameter update rules.

Optimizers only read ``.grad`` (populated by autograd) and mutate
``.data`` — they never compute gradients or touch the backend directly
(ADR-009).
"""

from onemoreepoch.core.parameter import Parameter
from onemoreepoch.exceptions import OptimizerError


class Optimizer:
    """Base class. Subclasses override ``_update_parameter`` only."""

    def __init__(self, parameters: list[Parameter], lr: float) -> None:
        if not parameters:
            raise OptimizerError("empty_params")
        if lr <= 0:
            raise OptimizerError("lr_invalid", value=lr)
        self.parameters = list(parameters)
        self.lr = lr

    def step(self) -> None:
        """Apply one update to every parameter that has a gradient."""
        for param in self.parameters:
            if param.grad is not None:
                self._update_parameter(param)

    def _update_parameter(self, param: Parameter) -> None:
        """Update a single parameter in place. Subclasses implement this."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Reset gradients of all managed parameters."""
        for param in self.parameters:
            param.zero_grad()
