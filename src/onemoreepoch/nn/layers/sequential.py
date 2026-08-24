"""Sequential container: chains modules in order."""

from onemoreepoch.core.module import Module
from onemoreepoch.core.tensor import Tensor


class Sequential(Module):
    """Runs child modules in the order they were passed."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self._ordered: list[Module] = []
        for index, module in enumerate(modules):
            setattr(self, str(index), module)  # registers via Module.__setattr__
            self._ordered.append(module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._ordered:
            x = module(x)
        return x

    def __len__(self) -> int:
        return len(self._ordered)

    def __getitem__(self, index: int) -> Module:
        return self._ordered[index]

    def __repr__(self) -> str:
        inner = ", ".join(repr(m) for m in self._ordered)
        return f"Sequential({inner})"
