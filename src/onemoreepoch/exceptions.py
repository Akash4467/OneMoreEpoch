from typing import Any

from onemoreepoch.messages import get_meme_for_key, get_message


# Base exception: resolves a message key + kwargs to personality-mode text and appends a meme if one matches
class OneMoreEpochError(Exception):
    def __init__(self, message_key: str, **fmt: Any) -> None:
        self.message_key = message_key
        text = get_message(message_key, **fmt)
        meme = get_meme_for_key(message_key)
        super().__init__(f"{text}\n\n{meme}" if meme else text)


# Base class for Tensor-level errors, not raised directly
class TensorError(OneMoreEpochError):
    pass


# Raised for incompatible shapes in an operation (matmul, broadcasting)
class ShapeError(TensorError, ValueError):
    pass


# Raised for incompatible or unsupported dtypes
class DTypeError(TensorError, TypeError):
    pass


# Raised for invalid autograd engine usage (e.g. backward() misuse)
class AutogradError(OneMoreEpochError, RuntimeError):
    pass


# Raised when a backend is unknown, unavailable, or misconfigured
class BackendError(OneMoreEpochError, RuntimeError):
    pass


# Raised for invalid Module usage (e.g. state_dict key/shape mismatch)
class ModuleError(OneMoreEpochError, ValueError):
    pass


# Raised for invalid optimizer configuration (bad lr, empty parameters)
class OptimizerError(OneMoreEpochError, ValueError):
    pass


# Raised for invalid dataset or data-loading configuration
class DataError(OneMoreEpochError, ValueError):
    pass


# Warning for gradient explosion/vanishing, never treated as an error
class GradientWarning(UserWarning):
    pass


__all__ = [
    "AutogradError",
    "BackendError",
    "DTypeError",
    "DataError",
    "GradientWarning",
    "ModuleError",
    "OneMoreEpochError",
    "OptimizerError",
    "ShapeError",
    "TensorError",
]
