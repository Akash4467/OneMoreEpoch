"""Adam: adaptive moment estimation."""

from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.parameter import Parameter
from onemoreepoch.exceptions import OptimizerError
from onemoreepoch.optim.optimizer import Optimizer


def _check_range(optimizer: str, param: str, value: float, low: float, high: float) -> None:
    if not low <= value < high:
        raise OptimizerError(
            "optimizer_param_invalid",
            optimizer=optimizer,
            param=param,
            value=value,
            constraint=f"{low} <= {param} < {high}",
        )


class Adam(Optimizer):
    """m_t, v_t bias-corrected moment estimates; param -= lr * m_hat / (sqrt(v_hat) + eps)."""

    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, lr)
        beta1, beta2 = betas
        _check_range("Adam", "beta1", beta1, 0.0, 1.0)
        _check_range("Adam", "beta2", beta2, 0.0, 1.0)
        if eps <= 0.0:
            raise OptimizerError(
                "optimizer_param_invalid",
                optimizer="Adam",
                param="eps",
                value=eps,
                constraint="eps > 0.0",
            )
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self._m: dict[int, object] = {}
        self._v: dict[int, object] = {}
        self._t: dict[int, int] = {}

    def _update_parameter(self, param: Parameter) -> None:
        key = id(param)
        grad = param.grad
        m = self._m.get(key, 0.0)
        v = self._v.get(key, 0.0)
        t = self._t.get(key, 0) + 1

        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
        self._m[key], self._v[key], self._t[key] = m, v, t

        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)
        # sqrt has no arithmetic-operator form; the one deliberate
        # backend call amid otherwise-raw `.data`/`.grad` arithmetic.
        param.data = param.data - self.lr * m_hat / (get_backend().sqrt(v_hat) + self.eps)
