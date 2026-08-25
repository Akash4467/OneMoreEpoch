from onemoreepoch.core.module import Module
from onemoreepoch.core.tensor import Tensor


# Container that runs child modules in the order they were passed
class Sequential(Module):
    # Registers each module under a numeric attribute name, in order
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self._ordered: list[Module] = []
        for index, module in enumerate(modules):
            setattr(self, str(index), module)
            self._ordered.append(module)

    # Chains x through every child module in order
    def forward(self, x: Tensor) -> Tensor:
        for module in self._ordered:
            x = module(x)
        return x

    # Returns the number of child modules
    def __len__(self) -> int:
        return len(self._ordered)

    # Returns the child module at index
    def __getitem__(self, index: int) -> Module:
        return self._ordered[index]

    # Returns a debug string representation
    def __repr__(self) -> str:
        inner = ", ".join(repr(m) for m in self._ordered)
        return f"Sequential({inner})"
