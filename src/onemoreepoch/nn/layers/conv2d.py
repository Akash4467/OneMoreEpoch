"""2-D convolution layer (im2col + matmul, correctness over speed)."""

from onemoreepoch.autograd.functions import Conv2DOp
from onemoreepoch.core.module import Module
from onemoreepoch.core.parameter import Parameter
from onemoreepoch.core.tensor import Tensor
from onemoreepoch.nn.init import kaiming_uniform_


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else tuple(value)


class Conv2D(Module):
    """Applies a 2-D convolution over a (N, C_in, H, W) input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw
        self.weight = kaiming_uniform_(
            Parameter(Tensor.zeros(out_channels, in_channels, kh, kw).data),
            fan_in=fan_in,
        )
        if bias:
            self.bias = kaiming_uniform_(
                Parameter(Tensor.zeros(out_channels).data), fan_in=fan_in
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = Conv2DOp.apply(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            out = out + self.bias.reshape(1, self.out_channels, 1, 1)
        return out

    def __repr__(self) -> str:
        return (
            f"Conv2D(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None})"
        )
