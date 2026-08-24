"""AdaGrad: divide the gradient by the square root of its running sum of squares."""

from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.parameter import Parameter
from onemoreepoch.exceptions import OptimizerError
from onemoreepoch.optim.optimizer import Optimizer


class AdaGrad(Optimizer):
    """G_t += g^2 (accumulated, never decayed); param -= lr * g / (sqrt(G_t) + eps)."""

    def __init__(
        self, parameters: list[Parameter], lr: float = 0.01, eps: float = 1e-10
    ) -> None:
        super().__init__(parameters, lr)
        if eps <= 0.0:
            raise OptimizerError(
                "optimizer_param_invalid",
                optimizer="AdaGrad",
                param="eps",
                value=eps,
                constraint="eps > 0.0",
            )
        self.eps = eps
        self._sum_sq: dict[int, object] = {}

    def _update_parameter(self, param: Parameter) -> None:
        key = id(param)
        grad = param.grad
        accumulated = self._sum_sq.get(key, 0.0) + grad * grad
        self._sum_sq[key] = accumulated
        param.data = param.data - self.lr * grad / (
            get_backend().sqrt(accumulated) + self.eps
        )
