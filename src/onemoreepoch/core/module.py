from collections.abc import Iterator
from typing import Any

from onemoreepoch.core.parameter import Parameter
from onemoreepoch.core.tensor import Tensor
from onemoreepoch.exceptions import ModuleError


# Base class for all neural network modules
class Module:
    # Initializes empty parameter/submodule registries and training mode
    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    # Auto-registers Parameter/Module values into their tracking dicts on assignment
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    # Computes the module's output; must be overridden by subclasses
    def forward(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement forward().")

    # Calls forward() on the given inputs
    def __call__(self, *inputs: Tensor) -> Tensor:
        return self.forward(*inputs)

    # Returns all trainable parameters in this module, recursively
    def parameters(self) -> list[Parameter]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    # Yields (dotted_name, parameter) pairs, recursively
    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            yield (f"{prefix}{name}", param)
        for name, module in self._modules.items():
            yield from module.named_parameters(prefix=f"{prefix}{name}.")

    # Resets gradients of all parameters
    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    # Sets this module and all submodules to training mode
    def train(self) -> "Module":
        object.__setattr__(self, "training", True)
        for module in self._modules.values():
            module.train()
        return self

    # Sets this module and all submodules to evaluation mode
    def eval(self) -> "Module":
        object.__setattr__(self, "training", False)
        for module in self._modules.values():
            module.eval()
        return self

    # Returns a flat mapping of dotted parameter names to raw arrays
    def state_dict(self) -> dict[str, Any]:
        return {name: param.data for name, param in self.named_parameters()}

    # Loads raw arrays into parameters by dotted name, validating keys and shapes
    def load_state_dict(self, state: dict[str, Any]) -> None:
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
