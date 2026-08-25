from typing import Any, Protocol, runtime_checkable


# Structural protocol describing the subset of Tensor that graph traversal touches
@runtime_checkable
class GraphNode(Protocol):
    creator: Any
    parents: tuple["GraphNode", ...]
    context: Any
    requires_grad: bool
    grad: Any
