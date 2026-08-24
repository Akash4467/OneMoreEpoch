"""z = conv2d(x, weight) — im2col + a single 2-D matmul, no bias.

Bias is deliberately not handled here; the ``nn.layers.Conv2D`` module
adds it as a plain broadcasted ``Tensor`` addition after this Function,
mirroring how ``Linear`` composes MatMul + Add instead of folding bias
into one Function.
"""

from typing import Any

from onemoreepoch.autograd.context import Context
from onemoreepoch.autograd.function import Function
from onemoreepoch.core.backend.registry import get_backend


class Conv2DOp(Function):
    """z = conv2d(x, weight), x: (N, C_in, H, W), weight: (C_out, C_in, KH, KW)"""

    @staticmethod
    def forward(
        ctx: Context,
        x: Any,
        weight: Any,
        *,
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> Any:
        backend = get_backend()
        n, c_in, h, w = x.shape
        c_out, _, kh, kw = weight.shape
        sh, sw = stride
        ph, pw = padding
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1
        k = c_in * kh * kw

        cols = backend.im2col(x, (kh, kw), (sh, sw), (ph, pw))  # (N, K, HW_out)
        weight_2d = backend.reshape(weight, (c_out, k))
        cols_2d = backend.reshape(
            backend.transpose(cols, (1, 0, 2)), (k, n * h_out * w_out)
        )
        out_2d = backend.matmul(weight_2d, cols_2d)  # (C_out, N*HW_out)
        out = backend.reshape(
            backend.transpose(
                backend.reshape(out_2d, (c_out, n, h_out * w_out)), (1, 0, 2)
            ),
            (n, c_out, h_out, w_out),
        )

        ctx.save_for_backward(weight_2d, cols_2d)
        ctx.extras.update(
            x_shape=x.shape,
            kernel_size=(kh, kw),
            stride=stride,
            padding=padding,
            c_out=c_out,
            k=k,
            n=n,
            h_out=h_out,
            w_out=w_out,
        )
        return out

    @staticmethod
    def backward(ctx: Context, grad: Any) -> tuple[Any, ...]:
        backend = get_backend()
        weight_2d, cols_2d = ctx.saved_tensors
        e = ctx.extras
        n, c_out, k, h_out, w_out = e["n"], e["c_out"], e["k"], e["h_out"], e["w_out"]
        kh, kw = e["kernel_size"]

        grad_2d = backend.reshape(
            backend.transpose(
                backend.reshape(grad, (n, c_out, h_out * w_out)), (1, 0, 2)
            ),
            (c_out, n * h_out * w_out),
        )
        grad_weight_2d = backend.matmul(
            grad_2d, backend.transpose(cols_2d)
        )  # (C_out, K)
        grad_cols_2d = backend.matmul(
            backend.transpose(weight_2d), grad_2d
        )  # (K, N*HW_out)
        grad_cols = backend.transpose(
            backend.reshape(grad_cols_2d, (k, n, h_out * w_out)), (1, 0, 2)
        )  # (N, K, HW_out)

        grad_x = backend.col2im(
            grad_cols, e["x_shape"], (kh, kw), e["stride"], e["padding"]
        )
        grad_weight = backend.reshape(grad_weight_2d, (c_out, e["x_shape"][1], kh, kw))
        return grad_x, grad_weight
