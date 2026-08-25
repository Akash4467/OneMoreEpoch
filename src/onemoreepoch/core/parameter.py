from typing import Any

from onemoreepoch.core.tensor import Tensor


# A trainable tensor stored on a module (requires_grad defaults to True)
class Parameter(Tensor):
    # Constructs the parameter as a Tensor with requires_grad defaulted to True
    def __init__(
        self, data: Any, *, dtype: Any = None, requires_grad: bool = True
    ) -> None:
        super().__init__(data, dtype=dtype, requires_grad=requires_grad)
