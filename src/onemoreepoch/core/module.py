"""Module: the abstract base class for all neural network components.

Handles parameter/submodule registration via ``__setattr__`` hooks,
train/eval mode switching, and state serialization. Concrete layers
subclass this and implement ``forward()``.
"""

from collections.abc import Iterator
from typing import Any

from onemoreepoch.core.parameter import Parameter
from onemoreepoch.core.tensor import Tensor
from onemoreepoch.exceptions import ModuleError


class Module:
    """Base class for all neural network modules."""

    def __init__(self) -> None:
        # object.__setattr__ because our __setattr__ reads these dicts.
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # Auto-register parameters and submodules on attribute assignment.
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    # -- forward ---------------------------------------------------------

    def forward(self, *inputs: Tensor) -> Tensor:
        """Compute the module's output. Must be overridden by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward().")

    def __call__(self, *inputs: Tensor) -> Tensor:
        return self.forward(*inputs)

    # -- parameter access ------------------------------------------------

    def parameters(self) -> list[Parameter]:
        """Return all trainable parameters in this module, recursively."""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Yield (dotted_name, parameter) pairs, recursively."""
        for name, param in self._parameters.items():
            yield (f"{prefix}{name}", param)
        for name, module in self._modules.items():
            yield from module.named_parameters(prefix=f"{prefix}{name}.")

    def zero_grad(self) -> None:
        """Reset gradients of all parameters."""
        for param in self.parameters():
            param.zero_grad()

    # -- mode switching --------------------------------------------------

    def train(self) -> "Module":
        """Set this module and all submodules to training mode."""
        object.__setattr__(self, "training", True)
        for module in self._modules.values():
            module.train()
        return self

    def eval(self) -> "Module":
        """Set this module and all submodules to evaluation mode."""
        object.__setattr__(self, "training", False)
        for module in self._modules.values():
            module.eval()
        return self

    # -- serialization ---------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a flat mapping of dotted parameter names to raw arrays."""
        return {name: param.data for name, param in self.named_parameters()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load raw arrays into parameters by dotted name."""
        own = dict(self.named_parameters())
        missing = own.keys() - state.keys()
        unexpected = state.keys() - own.keys()
        if missing or unexpected:
            raise ModuleError(
                "state_dict_key_mismatch",
                missing=sorted(missing),
                unexpected=sorted(unexpected),
            )
        for name, param in own.items():
            if tuple(state[name].shape) != param.shape:
                raise ModuleError(
                    "state_dict_shape_mismatch",
                    name=name,
                    expected=param.shape,
                    actual=tuple(state[name].shape),
                )
            param.data = state[name]
