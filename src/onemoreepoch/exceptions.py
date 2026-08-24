"""Custom exceptions for OneMoreEpoch.

Every exception resolves its text through ``messages.get_message()``
at raise time, so the active personality mode (classic/hindi/roast)
decides the wording. Each class also inherits the matching builtin
(``ValueError``/``RuntimeError``/``TypeError``) so existing ``except``
clauses and tests keep working unchanged.
"""

from typing import Any

from onemoreepoch.messages import get_message, get_meme_for_key


class OneMoreEpochError(Exception):
    """Base class for all OneMoreEpoch exceptions.

    Subclasses are raised with a message *key* plus format kwargs; the
    text is resolved from the active message mode. A matching meme may
    be appended (doc §20) — it decorates the message, never replaces
    it, and a meme lookup failure can never break exception
    construction (get_meme_for_key never raises).
    """

    def __init__(self, message_key: str, **fmt: Any) -> None:
        self.message_key = message_key
        text = get_message(message_key, **fmt)
        meme = get_meme_for_key(message_key)
        super().__init__(f"{text}\n\n{meme}" if meme else text)


class TensorError(OneMoreEpochError):
    """Base class for Tensor-level errors. Not raised directly."""


class ShapeError(TensorError, ValueError):
    """Incompatible shapes for an operation (matmul, broadcasting)."""


class DTypeError(TensorError, TypeError):
    """Incompatible or unsupported dtype for an operation."""


class AutogradError(OneMoreEpochError, RuntimeError):
    """Invalid use of the autograd engine (e.g. backward() misuse)."""


class BackendError(OneMoreEpochError, RuntimeError):
    """A backend is unknown, unavailable, or misconfigured."""


class ModuleError(OneMoreEpochError, ValueError):
    """Invalid Module usage (e.g. state_dict key/shape mismatch)."""


class OptimizerError(OneMoreEpochError, ValueError):
    """Invalid optimizer configuration (bad lr, empty parameters)."""


class DataError(OneMoreEpochError, ValueError):
    """Invalid dataset or data-loading configuration."""


class GradientWarning(UserWarning):
    """Gradient-health warning (explosion/vanishing), never an error."""


__all__ = [
    "AutogradError",
    "BackendError",
    "DataError",
    "DTypeError",
    "GradientWarning",
    "ModuleError",
    "OneMoreEpochError",
    "OptimizerError",
    "ShapeError",
    "TensorError",
]
