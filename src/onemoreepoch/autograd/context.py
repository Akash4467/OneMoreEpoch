"""Context: per-call storage for backward-pass intermediates.

A Context exists for exactly one Function call (ADR-006). Forward saves
whatever backward will need; nothing else persists on the Function.
"""

from typing import Any


class Context:
    """Short-lived container for one Function call's saved values."""

    def __init__(self) -> None:
        self._saved: tuple[Any, ...] = ()
        # Free-form extras (axis, shape, exponent, ...) set by forward().
        self.extras: dict[str, Any] = {}

    def save_for_backward(self, *values: Any) -> None:
        """Store raw arrays/values needed by backward()."""
        self._saved = values

    @property
    def saved_tensors(self) -> tuple[Any, ...]:
        """Return the values stored by save_for_backward()."""
        return self._saved
