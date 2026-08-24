"""Graph traversal: reverse topological ordering over GraphNode.parents.

Split out of ``engine.py`` so the DAG-ordering logic (doc §30's DAG
pattern) is independently testable and reusable without pulling in the
gradient-accumulation/warning logic that lives in the engine.
"""

from onemoreepoch.autograd.node import GraphNode


def topological_order(root: GraphNode) -> list[GraphNode]:
    """Return nodes in reverse topological order (root first).

    Iterative DFS post-order over ``parents`` pointers, reversed —
    guarantees every node is visited only after all its consumers.
    """
    order: list[GraphNode] = []
    visited: set[int] = set()
    stack: list[tuple[GraphNode, bool]] = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for parent in node.parents:
            if id(parent) not in visited:
                stack.append((parent, False))
    order.reverse()
    return order
