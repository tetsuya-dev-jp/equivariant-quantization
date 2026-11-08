"""Exact answer oracles.

All correctness labels used downstream are recomputed here from the saved
semantic structure and canonical option identities. Generated label metadata is
never trusted as the source of truth.
"""

from collections import deque
from typing import Any


def arithmetic_value(node: Any) -> int:
    """Evaluate a small integer expression tree."""
    if isinstance(node, int):
        return node
    if not isinstance(node, dict):
        raise ValueError(f"invalid arithmetic node: {node!r}")

    op = node["op"]
    left = arithmetic_value(node["left"])
    right = arithmetic_value(node["right"])
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    raise ValueError(f"unsupported arithmetic operator: {op!r}")


def boolean_value(node: Any, assignment: dict[str, bool]) -> bool:
    """Evaluate a Boolean expression under one complete assignment."""
    if isinstance(node, str):
        return assignment[node]
    if not isinstance(node, dict):
        raise ValueError(f"invalid Boolean node: {node!r}")

    op = node["op"]
    if op == "not":
        return not boolean_value(node["child"], assignment)
    if op == "and":
        return boolean_value(node["left"], assignment) and boolean_value(
            node["right"], assignment
        )
    if op == "or":
        return boolean_value(node["left"], assignment) or boolean_value(
            node["right"], assignment
        )
    raise ValueError(f"unsupported Boolean operator: {op!r}")


def graph_distances(structure: dict[str, Any]) -> dict[str, int]:
    """Shortest directed-edge distances from the declared source node."""
    nodes = structure["nodes"]
    adjacency = {node: [] for node in nodes}
    for source, target in structure["edges"]:
        adjacency[source].append(target)

    source_node = structure["source"]
    distances = {source_node: 0}
    queue = deque([source_node])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances


def graph_node_satisfies(node: str, structure: dict[str, Any]) -> bool:
    """Whether one candidate node satisfies the declared graph relation."""
    source_node = structure["source"]
    if node == source_node:
        return False

    relation = structure["relation"]
    distances = graph_distances(structure)
    if relation["type"] == "reachable":
        return node in distances
    if relation["type"] == "distance":
        return distances.get(node) == relation["distance"]
    raise ValueError(f"unsupported graph relation: {relation!r}")


def canonical_answer(
    task_family: str,
    structure: dict[str, Any],
    canonical_options: list[Any],
) -> int:
    """Return the unique canonical-option index that satisfies the question.

    Raises ValueError when the stored options do not have exactly one correct
    answer, which would indicate a generator bug rather than a model failure.
    """
    if task_family == "arithmetic":
        target = arithmetic_value(structure["expression"])
        matches = [
            index
            for index, option in enumerate(canonical_options)
            if option == target
        ]
    elif task_family == "boolean":
        variables = structure["variables"]
        matches = []
        for index, option in enumerate(canonical_options):
            assignment = dict(zip(variables, option, strict=True))
            if boolean_value(structure["expression"], assignment):
                matches.append(index)
    elif task_family == "graph":
        matches = [
            index
            for index, option in enumerate(canonical_options)
            if graph_node_satisfies(option, structure)
        ]
    else:
        raise ValueError(f"unsupported task family: {task_family!r}")

    if len(matches) != 1:
        raise ValueError(
            "canonical options must contain exactly one correct answer; "
            f"found {len(matches)} for family {task_family!r}"
        )
    return matches[0]
