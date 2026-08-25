from onemoreepoch.core.parameter import Parameter
from onemoreepoch.exceptions import OptimizerError
from onemoreepoch.optim.optimizer import Optimizer


# Stochastic gradient descent with optional momentum
class SGD(Optimizer):
    # Validates momentum and initializes velocity state
    def __init__(
        self, parameters: list[Parameter], lr: float = 0.01, momentum: float = 0.0
    ) -> None:
        super().__init__(parameters, lr)
        if not 0.0 <= momentum < 1.0:
            raise OptimizerError(
                "optimizer_param_invalid",
                optimizer="SGD",
                param="momentum",
                value=momentum,
                constraint="0.0 <= momentum < 1.0",
            )
        self.momentum = momentum
        self._velocity: dict[int, object] = {}

    # Updates the parameter using momentum-accumulated gradient descent
    def _update_parameter(self, param: Parameter) -> None:
        update = param.grad
        if self.momentum > 0.0:
            velocity = self._velocity.get(id(param))
            velocity = (
                param.grad
                if velocity is None
                else self.momentum * velocity + param.grad
            )
            self._velocity[id(param)] = velocity
            update = velocity
        param.data = param.data - self.lr * update
