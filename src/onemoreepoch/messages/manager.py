"""Personality-driven messaging for errors and training output.

Resolves message templates from the active mode (see ``config``) and
formats them. Per ADR-010 this module only holds/selects and formats
text — it never raises domain exceptions or performs numerical
computation. Exceptions that *use* these messages live in
``onemoreepoch.exceptions``.

Also the facade onto the meme subsystem (doc §20): ``get_meme`` and
``get_meme_for_key`` never raise — a meme failure must never break a
Tensor operation, training step, or technical exception.
"""

from onemoreepoch import config
from onemoreepoch.messages import classic, hindi, roast
from onemoreepoch.messages.memes.selector import MemeSelector

_MODES = {
    "classic": classic,
    "hindi": hindi,
    "roast": roast,
}

# Maps a subset of message keys onto doc §20's fixed meme categories.
# Keys with no natural category (e.g. ModuleError/DataError variants)
# are simply absent — get_meme_for_key returns None for those, no meme.
_MESSAGE_KEY_CATEGORIES: dict[str, str] = {
    "shape_mismatch_matmul": "shape_error",
    "broadcast_failure": "shape_error",
    "backward_no_grad": "gradient_error",
    "backward_non_scalar": "gradient_error",
    "gradient_explosion": "gradient_error",
    "vanishing_gradient": "gradient_error",
    "nan_loss": "nan_error",
    "loss_increasing": "general_warning",
    "lr_invalid": "optimizer_error",
    "empty_params": "optimizer_error",
    "optimizer_param_invalid": "optimizer_error",
    "unknown_backend": "device_error",
    "rust_backend_unavailable": "device_error",
    "training_complete": "training_complete",
}

_selector = MemeSelector()


def get_message(key: str, **fmt: object) -> str:
    """Return the template for ``key`` in the active mode, formatted.

    Falls back to classic if the active mode is missing the key, so a
    fun mode can never make an error message disappear.
    """
    module = _MODES[config.get_message_mode()]
    template = module.MESSAGES.get(key, classic.MESSAGES[key])
    return template.format(**fmt)


def get_banter(index: int) -> str:
    """Return an epoch banter line, rotating deterministically."""
    lines = _MODES[config.get_message_mode()].EPOCH_BANTER
    return lines[index % len(lines)]


def get_meme(category: str | None) -> str | None:
    """Return a meme line for ``category`` in the active mode, or None.

    Swallows every failure (missing catalog, bad category, whatever) —
    per doc §26/§20 a meme problem must never propagate.
    """
    if not category:
        return None
    try:
        return _selector.select(category, config.get_message_mode())
    except Exception:  # noqa: BLE001 - a meme failure must never propagate
        return None


def get_meme_for_key(message_key: str) -> str | None:
    """Look up the meme category for a message key, then get_meme() it."""
    return get_meme(_MESSAGE_KEY_CATEGORIES.get(message_key))


__all__ = ["get_banter", "get_meme", "get_meme_for_key", "get_message"]
