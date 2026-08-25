from typing import Any


# Short-lived container holding one Function call's saved values for backward
class Context:
    # Initializes empty saved-values and extras storage
    def __init__(self) -> None:
        self._saved: tuple[Any, ...] = ()
        self.extras: dict[str, Any] = {}

    # Stores raw arrays/values that backward() will need
    def save_for_backward(self, *values: Any) -> None:
        self._saved = values

    # Returns the values stored by save_for_backward()
    @property
    def saved_tensors(self) -> tuple[Any, ...]:
        return self._saved
