from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.parameter import Parameter
from onemoreepoch.exceptions import OptimizerError
from onemoreepoch.optim.adam import _check_range
from onemoreepoch.optim.optimizer import Optimizer


# Divides the gradient by a running RMS of recent squared gradients
class RMSProp(Optimizer):
    # Validates alpha/momentum/eps and initializes running-average state
    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        _check_range("RMSProp", "alpha", alpha, 0.0, 1.0)
        _check_range("RMSProp", "momentum", momentum, 0.0, 1.0)
        if eps <= 0.0:
            raise OptimizerError(
                "optimizer_param_invalid",
                optimizer="RMSProp",
                param="eps",
                value=eps,
                constraint="eps > 0.0",
            )
        self.alpha, self.eps, self.momentum = alpha, eps, momentum
        self._v: dict[int, object] = {}
        self._velocity: dict[int, object] = {}

    # Updates the parameter using an RMS-normalized gradient, with optional momentum
    def _update_parameter(self, param: Parameter) -> None:
        key = id(param)
        grad = param.grad
        v = self._v.get(key, 0.0)
        v = self.alpha * v + (1 - self.alpha) * (grad * grad)
        self._v[key] = v

        update = grad / (get_backend().sqrt(v) + self.eps)
        if self.momentum > 0.0:
            velocity = self._velocity.get(key)
            velocity = update if velocity is None else self.momentum * velocity + update
            self._velocity[key] = velocity
            update = velocity
        param.data = param.data - self.lr * update
