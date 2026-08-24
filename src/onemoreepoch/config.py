"""Global configuration and runtime settings.

Holds process-wide toggles: which message personality is active
(ADR-010 — fun modes are opt-in, professional is the default) and
whether the autograd engine runs extra gradient-health diagnostics.

This module must stay dependency-free within onemoreepoch so any
package may import it without creating cycles.
"""

import os

MESSAGE_MODES = ("classic", "hindi", "roast")

_message_mode = "classic"
_debug_checks = False


def get_message_mode() -> str:
    """Return the active message personality mode."""
    return _message_mode


def set_message_mode(mode: str) -> None:
    """Set the active message personality mode.

    Valid modes: ``classic`` (professional, default), ``hindi``,
    ``roast``.
    """
    global _message_mode
    if mode not in MESSAGE_MODES:
        raise ValueError(
            f"Unknown message mode: {mode!r}. Valid modes: {MESSAGE_MODES}."
        )
    _message_mode = mode


def debug_checks_enabled() -> bool:
    """Return whether gradient-health diagnostics are active."""
    return _debug_checks


def set_debug_checks(enabled: bool) -> None:
    """Enable/disable gradient explosion/vanishing warnings in backward."""
    global _debug_checks
    _debug_checks = bool(enabled)


def _init_from_env() -> None:
    """Read initial settings from environment variables.

    ``ONEMOREEPOCH_MESSAGES=classic|hindi|roast`` picks the mode;
    ``EDUCATIONAL_MODE=1`` (JOURNEY.md) is a shorthand for hindi.
    Invalid values are ignored — config must never crash an import.
    """
    global _message_mode
    env_mode = os.environ.get("ONEMOREEPOCH_MESSAGES", "").strip().lower()
    if env_mode in MESSAGE_MODES:
        _message_mode = env_mode
    elif os.environ.get("EDUCATIONAL_MODE", "").strip() == "1":
        _message_mode = "hindi"


_init_from_env()
