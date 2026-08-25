from onemoreepoch.core.backend.registry import get_backend
from onemoreepoch.core.module import Module
from onemoreepoch.core.parameter import Parameter
from onemoreepoch.core.tensor import Tensor
from onemoreepoch.exceptions import ModuleError
from onemoreepoch.nn.init import ones_, zeros_


# Normalizes a (N, C) input to zero mean / unit variance per feature across the batch
class BatchNorm(Module):
    # Validates eps/momentum and builds gamma/beta parameters plus running-stat buffers
    def __init__(
        self, num_features: int, *, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ModuleError(
                "module_param_invalid",
                module="BatchNorm",
                param="eps",
                value=eps,
                constraint="eps > 0.0",
            )
        if not 0.0 <= momentum <= 1.0:
            raise ModuleError(
                "module_param_invalid",
                module="BatchNorm",
                param="momentum",
                value=momentum,
                constraint="0.0 <= momentum <= 1.0",
            )
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = ones_(Parameter(Tensor.zeros(num_features).data))
        self.beta = zeros_(Parameter(Tensor.zeros(num_features).data))
        backend = get_backend()
        self.running_mean = backend.zeros((num_features,))
        self.running_var = backend.ones((num_features,))

    # Normalizes x using batch statistics (training) or running statistics (eval)
    def forward(self, x: Tensor) -> Tensor:
        backend = get_backend()
        if self.training:
            batch_mean = x.mean(axis=0, keepdims=True)
            centered = x - batch_mean
            batch_var = (centered * centered).mean(axis=0, keepdims=True)

            flat_mean = backend.reshape(batch_mean.data, (self.num_features,))
            flat_var = backend.reshape(batch_var.data, (self.num_features,))
            self.running_mean = backend.add(
                backend.multiply(self.running_mean, 1.0 - self.momentum),
                backend.multiply(flat_mean, self.momentum),
            )
            self.running_var = backend.add(
                backend.multiply(self.running_var, 1.0 - self.momentum),
                backend.multiply(flat_var, self.momentum),
            )
            mean, var = batch_mean, batch_var
        else:
            mean = Tensor(backend.reshape(self.running_mean, (1, self.num_features)))
            var = Tensor(backend.reshape(self.running_var, (1, self.num_features)))

        normalized = (x - mean) / ((var + self.eps) ** 0.5)
        return normalized * self.gamma + self.beta

    # Returns a debug string representation
    def __repr__(self) -> str:
        return f"BatchNorm(num_features={self.num_features}, eps={self.eps})"
