"""GraphNode: the structural shape autograd needs from a graph node.

A ``typing.Protocol`` rather than importing ``core.tensor.Tensor``
directly — ``autograd`` depends on this abstract shape, not on a
concrete ``core`` type, matching the doc's dependency-inversion intent
(§29, §38). ``Tensor`` satisfies this structurally; no inheritance
needed.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphNode(Protocol):
    """The subset of Tensor that graph traversal actually touches."""

    creator: Any
    parents: tuple["GraphNode", ...]
    context: Any
    requires_grad: bool
    grad: Any
