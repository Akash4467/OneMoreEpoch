from onemoreepoch import config
from onemoreepoch.messages import classic, hindi, roast
from onemoreepoch.messages.memes.selector import MemeSelector

_MODES = {
    "classic": classic,
    "hindi": hindi,
    "roast": roast,
}

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


# Returns the formatted message template for key in the active mode
def get_message(key: str, **fmt: object) -> str:
    module = _MODES[config.get_message_mode()]
    template = module.MESSAGES.get(key, classic.MESSAGES[key])
    return template.format(**fmt)


# Returns an epoch banter line, rotating deterministically
def get_banter(index: int) -> str:
    lines = _MODES[config.get_message_mode()].EPOCH_BANTER
    return lines[index % len(lines)]


# Returns a meme line for category in the active mode, or None on any failure
def get_meme(category: str | None) -> str | None:
    if not category:
        return None
    try:
        return _selector.select(category, config.get_message_mode())
    except Exception:  # noqa: BLE001
        return None


# Looks up the meme category for a message key, then calls get_meme() on it
def get_meme_for_key(message_key: str) -> str | None:
    return get_meme(_MESSAGE_KEY_CATEGORIES.get(message_key))


__all__ = ["get_banter", "get_meme", "get_meme_for_key", "get_message"]
