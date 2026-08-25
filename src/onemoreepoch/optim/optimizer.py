from onemoreepoch.core.parameter import Parameter
from onemoreepoch.exceptions import OptimizerError


# Template-method base class for parameter update rules
class Optimizer:
    # Validates parameters/lr and stores them
    def __init__(self, parameters: list[Parameter], lr: float) -> None:
        if not parameters:
            raise OptimizerError("empty_params")
        if lr <= 0:
            raise OptimizerError("lr_invalid", value=lr)
        self.parameters = list(parameters)
        self.lr = lr

    # Applies one update to every parameter that has a gradient
    def step(self) -> None:
        for param in self.parameters:
            if param.grad is not None:
                self._update_parameter(param)

    # Updates a single parameter in place; subclasses implement this
    def _update_parameter(self, param: Parameter) -> None:
        raise NotImplementedError

    # Resets gradients of all managed parameters
    def zero_grad(self) -> None:
        for param in self.parameters:
            param.zero_grad()
