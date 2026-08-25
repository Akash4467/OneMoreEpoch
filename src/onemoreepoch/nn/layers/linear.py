from onemoreepoch.core.module import Module
from onemoreepoch.core.parameter import Parameter
from onemoreepoch.core.tensor import Tensor
from onemoreepoch.nn.init import kaiming_uniform_


# Fully connected layer: y = x @ W + b
class Linear(Module):
    # Builds weight/bias parameters with Kaiming-uniform initialization
    def __init__(
        self, in_features: int, out_features: int, *, bias: bool = True
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = kaiming_uniform_(
            Parameter(Tensor.zeros(in_features, out_features).data),
            fan_in=in_features,
        )
        if bias:
            self.bias = kaiming_uniform_(
                Parameter(Tensor.zeros(out_features).data), fan_in=in_features
            )
        else:
            self.bias = None

    # Computes x @ weight (+ bias)
    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    # Returns a debug string representation
    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )
