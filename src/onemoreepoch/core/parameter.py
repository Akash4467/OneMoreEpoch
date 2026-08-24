"""Trainable parameter wrappers."""

from typing import Any

from onemoreepoch.core.tensor import Tensor


class Parameter(Tensor):
    """A trainable tensor stored on a module."""

    def __init__(
        self, data: Any, *, dtype: Any = None, requires_grad: bool = True
    ) -> None:
        super().__init__(data, dtype=dtype, requires_grad=requires_grad)
