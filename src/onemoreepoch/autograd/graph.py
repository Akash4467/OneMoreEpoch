from onemoreepoch.autograd.node import GraphNode


# Returns nodes in reverse topological order (root first) via iterative DFS
def topological_order(root: GraphNode) -> list[GraphNode]:
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
