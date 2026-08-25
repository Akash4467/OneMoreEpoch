import os

MESSAGE_MODES = ("classic", "hindi", "roast")

_message_mode = "classic"
_debug_checks = False


# Returns the currently active message personality mode
def get_message_mode() -> str:
    return _message_mode


# Sets the active message personality mode, validating it's a known mode
def set_message_mode(mode: str) -> None:
    global _message_mode
    if mode not in MESSAGE_MODES:
        raise ValueError(
            f"Unknown message mode: {mode!r}. Valid modes: {MESSAGE_MODES}."
        )
    _message_mode = mode


# Returns whether gradient-health diagnostics are currently enabled
def debug_checks_enabled() -> bool:
    return _debug_checks


# Enables or disables gradient explosion/vanishing warnings during backward
def set_debug_checks(enabled: bool) -> None:
    global _debug_checks
    _debug_checks = bool(enabled)


# Reads ONEMOREEPOCH_MESSAGES / EDUCATIONAL_MODE env vars to set the initial mode
def _init_from_env() -> None:
    global _message_mode
    env_mode = os.environ.get("ONEMOREEPOCH_MESSAGES", "").strip().lower()
    if env_mode in MESSAGE_MODES:
        _message_mode = env_mode
    elif os.environ.get("EDUCATIONAL_MODE", "").strip() == "1":
        _message_mode = "hindi"


_init_from_env()
