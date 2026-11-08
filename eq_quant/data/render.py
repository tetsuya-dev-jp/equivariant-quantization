"""Prompt rendering for each task family.

Rendering is presentation only. It applies option order, entity renaming, and
independent-fact order without changing the canonical semantic structure.
"""

from typing import Any

from .schema import OPTION_LABELS


def render_arithmetic_expression(node: Any) -> str:
    if isinstance(node, int):
        return str(node)
    left = render_arithmetic_expression(node["left"])
    right = render_arithmetic_expression(node["right"])
    return f"({left} {node['op']} {right})"


def render_boolean_expression(node: Any, names: dict[str, str]) -> str:
    if isinstance(node, str):
        return names[node]
    if node["op"] == "not":
        return f"(not {render_boolean_expression(node['child'], names)})"
    left = render_boolean_expression(node["left"], names)
    right = render_boolean_expression(node["right"], names)
    return f"({left} {node['op']} {right})"


def _identity_names(items: list[str]) -> dict[str, str]:
    return {item: item for item in items}


def _display_names(
    canonical_names: list[str],
    rename: dict[str, Any] | None,
) -> dict[str, str]:
    if rename is None:
        return _identity_names(canonical_names)
    return dict(rename["canonical_to_display"])


def _option_lines(
    labels: tuple[str, ...],
    option_order: list[int],
    rendered_options: list[str],
) -> list[str]:
    return [
        f"{label}. {rendered_options[index]}"
        for label, index in zip(labels, option_order, strict=True)
    ]


def render_arithmetic_prompt(
    structure: dict[str, Any],
    canonical_options: list[Any],
    option_order: list[int],
    rename: dict[str, Any] | None,
    order: dict[str, Any] | None,
) -> str:
    if order is not None:
        raise ValueError("arithmetic order transformation is unavailable")

    names = _display_names([structure["name"]], rename)
    name = names[structure["name"]]
    expression = render_arithmetic_expression(structure["expression"])
    rendered = [str(option) for option in canonical_options]
    lines = [f"{name} calculates {expression}. What is the result?", ""]
    lines.extend(_option_lines(OPTION_LABELS, option_order, rendered))
    lines.extend(["", "Answer with A, B, C, or D."])
    return "\n".join(lines)


def render_boolean_prompt(
    structure: dict[str, Any],
    canonical_options: list[Any],
    option_order: list[int],
    rename: dict[str, Any] | None,
    order: dict[str, Any] | None,
) -> str:
    variables = structure["variables"]
    names = _display_names(variables, rename)
    if order is None:
        variable_order = list(range(len(variables)))
    else:
        variable_order = list(order["variable_order"])

    displayed_variables = [names[variables[index]] for index in variable_order]
    expression = render_boolean_expression(structure["expression"], names)

    rendered_options: list[str] = []
    for option in canonical_options:
        facts = [
            f"{names[variables[index]]}={option[index]}"
            for index in variable_order
        ]
        rendered_options.append(", ".join(facts))

    lines = [
        f"Variables: {', '.join(displayed_variables)}. Each is True or False.",
        f"Expression: {expression}",
        "Which assignment makes the expression True?",
        "",
    ]
    lines.extend(_option_lines(OPTION_LABELS, option_order, rendered_options))
    lines.extend(["", "Answer with A, B, C, or D."])
    return "\n".join(lines)


def render_graph_prompt(
    structure: dict[str, Any],
    canonical_options: list[Any],
    option_order: list[int],
    rename: dict[str, Any] | None,
    order: dict[str, Any] | None,
) -> str:
    nodes = structure["nodes"]
    names = _display_names(nodes, rename)
    if order is None:
        edge_order = list(range(len(structure["edges"])))
    else:
        edge_order = list(order["edge_order"])

    edge_texts = [
        f"{names[structure['edges'][index][0]]}->{names[structure['edges'][index][1]]}"
        for index in edge_order
    ]
    source = names[structure["source"]]
    relation = structure["relation"]
    if relation["type"] == "reachable":
        question = f"Starting from {source}, which node is reachable by a directed path?"
    elif relation["type"] == "distance":
        question = (
            f"Starting from {source}, which node has a shortest directed path "
            f"of length exactly {relation['distance']}?"
        )
    else:
        raise ValueError(f"unsupported graph relation: {relation!r}")

    rendered = [names[node] for node in canonical_options]
    lines = [
        f"Directed edges: {', '.join(edge_texts)}.",
        question,
        "",
    ]
    lines.extend(_option_lines(OPTION_LABELS, option_order, rendered))
    lines.extend(["", "Answer with A, B, C, or D."])
    return "\n".join(lines)


def render_prompt(
    task_family: str,
    structure: dict[str, Any],
    canonical_options: list[Any],
    option_order: list[int],
    rename: dict[str, Any] | None,
    order: dict[str, Any] | None,
) -> str:
    if task_family == "arithmetic":
        return render_arithmetic_prompt(
            structure, canonical_options, option_order, rename, order
        )
    if task_family == "boolean":
        return render_boolean_prompt(
            structure, canonical_options, option_order, rename, order
        )
    if task_family == "graph":
        return render_graph_prompt(
            structure, canonical_options, option_order, rename, order
        )
    raise ValueError(f"unsupported task family: {task_family!r}")
