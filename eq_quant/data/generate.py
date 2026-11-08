"""Deterministic procedural generators for the three task families."""

import hashlib
import itertools
import json
import random
from typing import Any

from .oracles import arithmetic_value, boolean_value, canonical_answer, graph_distances
from .render import render_prompt
from .schema import (
    OPTION_LABELS,
    TRANSFORM_AVAILABILITY,
    BaseInstance,
    Record,
)

ARITHMETIC_NAMES = (
    "Alice",
    "Ben",
    "Carla",
    "Diego",
    "Emma",
    "Farah",
    "Hiro",
    "Ines",
    "Jonas",
    "Keiko",
)

BOOLEAN_VARIABLES = ("p", "q", "r", "s", "t")
BOOLEAN_RENAME_POOL = ("m", "n", "u", "v", "w", "x", "y", "z", "a", "b", "c", "d")

# Node labels avoid A-D so they cannot be confused with option labels.
GRAPH_NODE_POOL = (
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
)
GRAPH_RENAME_POOL = (
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "O",
)


def derive_seed(*parts: Any) -> int:
    """Stable seed derivation independent of Python's randomized hash."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def semantic_key(task_family: str, structure: dict[str, Any]) -> str:
    """Exact structural identity, excluding option choices and proper names.

    The equivalence boundary is the normalized stored problem structure:
    arithmetic expression tree; Boolean expression tree with canonical variable
    names and variable count; or graph node set, directed edges, source, and
    relation. This is not full Boolean logical equivalence or graph isomorphism.
    Distractor choices and the arithmetic character name are presentation, so
    they are excluded.
    """
    if task_family == "arithmetic":
        payload = {"expression": structure["expression"]}
    elif task_family == "boolean":
        payload = {
            "expression": structure["expression"],
            "variable_count": structure["variable_count"],
        }
    elif task_family == "graph":
        payload = {
            "nodes": structure["nodes"],
            "edges": sorted([list(edge) for edge in structure["edges"]]),
            "source": structure["source"],
            "relation": structure["relation"],
        }
    else:
        raise ValueError(f"unsupported task family: {task_family!r}")
    return task_family + "|" + json.dumps(payload, sort_keys=True)


def _sample_arithmetic_expression(rng: random.Random, operation_count: int) -> Any:
    leaves = [rng.randint(1, 9)]
    for _ in range(operation_count):
        index = rng.randrange(len(leaves))
        old_leaf = leaves[index]
        new_leaf = rng.randint(1, 9)
        op = rng.choice(("+", "-", "*"))
        if rng.random() < 0.5:
            node = {"op": op, "left": old_leaf, "right": new_leaf}
        else:
            node = {"op": op, "left": new_leaf, "right": old_leaf}
        leaves[index : index + 1] = [node]
    return leaves[0]


def _arithmetic_distractors(rng: random.Random, correct: int) -> list[int]:
    candidates = [
        correct + offset
        for offset in (-4, -3, -2, -1, 1, 2, 3, 4)
    ]
    rng.shuffle(candidates)
    chosen: list[int] = []
    for value in candidates:
        if value != correct and value not in chosen:
            chosen.append(value)
        if len(chosen) == 3:
            return chosen
    raise RuntimeError("could not construct three distinct arithmetic distractors")


def generate_arithmetic_base(
    base_id: str,
    split: str,
    seed: int,
    structural: bool,
) -> BaseInstance:
    rng = random.Random(seed)
    operation_count = rng.choice((5, 6)) if structural else rng.choice((2, 3))
    for _ in range(200):
        expression = _sample_arithmetic_expression(rng, operation_count)
        value = arithmetic_value(expression)
        if abs(value) <= (2000 if structural else 200):
            break
    else:
        raise RuntimeError("could not generate a bounded arithmetic expression")

    distractors = _arithmetic_distractors(rng, value)
    canonical_options = [value, *distractors]
    rng.shuffle(canonical_options)

    structure = {
        "name": rng.choice(ARITHMETIC_NAMES),
        "expression": expression,
        "operation_count": operation_count,
    }
    return BaseInstance(
        base_id=base_id,
        split=split,
        task_family="arithmetic",
        structure=structure,
        canonical_options=canonical_options,
        semantic_key=semantic_key("arithmetic", structure),
    )


def _random_boolean_expression(
    rng: random.Random,
    variables: tuple[str, ...],
    depth: int,
) -> Any:
    if depth <= 0 or rng.random() < 0.35:
        return rng.choice(variables)
    if rng.random() < 0.25:
        return {
            "op": "not",
            "child": _random_boolean_expression(rng, variables, depth - 1),
        }
    return {
        "op": rng.choice(("and", "or")),
        "left": _random_boolean_expression(rng, variables, depth - 1),
        "right": _random_boolean_expression(rng, variables, depth - 1),
    }


def _all_assignments(variable_count: int) -> list[tuple[bool, ...]]:
    return list(itertools.product((False, True), repeat=variable_count))


def _boolean_expression_variables(node: Any) -> set[str]:
    if isinstance(node, str):
        return {node}
    if node["op"] == "not":
        return _boolean_expression_variables(node["child"])
    return _boolean_expression_variables(
        node["left"]
    ) | _boolean_expression_variables(node["right"])


def generate_boolean_base(
    base_id: str,
    split: str,
    seed: int,
    structural: bool,
) -> BaseInstance:
    rng = random.Random(seed)
    variable_count = 5 if structural else rng.choice((3, 4))
    depth = 3 if structural else 2
    variables = BOOLEAN_VARIABLES[:variable_count]

    for _ in range(500):
        expression = _random_boolean_expression(rng, variables, depth)
        if len(_boolean_expression_variables(expression)) < 2:
            continue
        assignments = _all_assignments(variable_count)
        values = [
            boolean_value(expression, dict(zip(variables, assignment, strict=True)))
            for assignment in assignments
        ]
        correct_assignments = [
            assignment for assignment, value in zip(assignments, values, strict=True)
            if value
        ]
        wrong_assignments = [
            assignment for assignment, value in zip(assignments, values, strict=True)
            if not value
        ]
        if correct_assignments and len(wrong_assignments) >= 3:
            break
    else:
        raise RuntimeError("could not generate a nonconstant Boolean expression")

    correct = rng.choice(correct_assignments)
    distractors = rng.sample(wrong_assignments, 3)
    canonical_options = [correct, *distractors]
    rng.shuffle(canonical_options)

    structure = {
        "variables": list(variables),
        "variable_count": variable_count,
        "expression": expression,
        "depth": depth,
    }
    return BaseInstance(
        base_id=base_id,
        split=split,
        task_family="boolean",
        structure=structure,
        canonical_options=canonical_options,
        semantic_key=semantic_key("boolean", structure),
    )


def _random_dag(rng: random.Random, node_count: int) -> tuple[list[str], list[list[str]]]:
    nodes = list(GRAPH_NODE_POOL[:node_count])
    topological_order = nodes[:]
    rng.shuffle(topological_order)
    edges: list[list[str]] = []
    for left in range(node_count):
        for right in range(left + 1, node_count):
            if rng.random() < 0.28:
                edges.append([topological_order[left], topological_order[right]])
    return nodes, edges


def generate_graph_base(
    base_id: str,
    split: str,
    seed: int,
    structural: bool,
) -> BaseInstance:
    rng = random.Random(seed)
    node_count = rng.choice((8, 9, 10)) if structural else rng.choice((5, 6, 7))

    for _ in range(500):
        nodes, edges = _random_dag(rng, node_count)
        if len(edges) < 3:
            continue

        source = rng.choice(nodes)
        relation_type = rng.choice(("reachable", "distance"))
        if relation_type == "reachable":
            relation = {"type": "reachable"}
        else:
            relation = {"type": "distance", "distance": rng.choice((1, 2))}

        structure = {
            "nodes": nodes,
            "edges": edges,
            "source": source,
            "relation": relation,
        }
        distances = graph_distances(structure)
        candidate_nodes = [node for node in nodes if node != source]
        satisfying: list[str] = []
        non_satisfying: list[str] = []
        for node in candidate_nodes:
            if relation_type == "reachable":
                satisfies = node in distances
            else:
                satisfies = distances.get(node) == relation["distance"]
            if satisfies:
                satisfying.append(node)
            else:
                non_satisfying.append(node)

        if len(satisfying) >= 1 and len(non_satisfying) >= 3:
            correct = rng.choice(satisfying)
            distractors = rng.sample(non_satisfying, 3)
            canonical_options = [correct, *distractors]
            rng.shuffle(canonical_options)
            break
    else:
        raise RuntimeError("could not generate a graph with exactly one correct candidate")

    return BaseInstance(
        base_id=base_id,
        split=split,
        task_family="graph",
        structure=structure,
        canonical_options=canonical_options,
        semantic_key=semantic_key("graph", structure),
    )


def generate_base(
    task_family: str,
    base_id: str,
    split: str,
    seed: int,
    structural: bool,
) -> BaseInstance:
    if task_family == "arithmetic":
        return generate_arithmetic_base(base_id, split, seed, structural)
    if task_family == "boolean":
        return generate_boolean_base(base_id, split, seed, structural)
    if task_family == "graph":
        return generate_graph_base(base_id, split, seed, structural)
    raise ValueError(f"unsupported task family: {task_family!r}")


def _rename_map(
    task_family: str,
    rng: random.Random,
    structure: dict[str, Any],
) -> dict[str, Any]:
    if task_family == "arithmetic":
        old_name = structure["name"]
        choices = [name for name in ARITHMETIC_NAMES if name != old_name]
        return {"canonical_to_display": {old_name: rng.choice(choices)}}

    if task_family == "boolean":
        canonical = structure["variables"]
        pool = [name for name in BOOLEAN_RENAME_POOL if name not in canonical]
        replacements = rng.sample(pool, len(canonical))
        return {"canonical_to_display": dict(zip(canonical, replacements, strict=True))}

    if task_family == "graph":
        canonical = structure["nodes"]
        pool = [name for name in GRAPH_RENAME_POOL if name not in canonical]
        replacements = rng.sample(pool, len(canonical))
        return {"canonical_to_display": dict(zip(canonical, replacements, strict=True))}

    raise ValueError(f"unsupported task family: {task_family!r}")


def _order_permutation(
    task_family: str,
    rng: random.Random,
    structure: dict[str, Any],
) -> dict[str, Any]:
    if task_family == "boolean":
        size = len(structure["variables"])
        key = "variable_order"
    elif task_family == "graph":
        size = len(structure["edges"])
        key = "edge_order"
    else:
        raise ValueError(f"order transformation unavailable for {task_family!r}")

    permutation = list(range(size))
    rng.shuffle(permutation)
    if permutation == list(range(size)):
        permutation[0], permutation[1] = permutation[1], permutation[0]
    return {key: permutation}


def _nonidentity_option_order(rng: random.Random) -> list[int]:
    order = list(range(len(OPTION_LABELS)))
    rng.shuffle(order)
    if order == list(range(len(OPTION_LABELS))):
        order[0], order[1] = order[1], order[0]
    return order


def build_records(base: BaseInstance) -> list[Record]:
    records: list[Record] = []
    semantic_correct = canonical_answer(
        base.task_family, base.structure, base.canonical_options
    )

    for transform in TRANSFORM_AVAILABILITY[base.task_family]:
        rng = random.Random(derive_seed(base.base_id, transform))
        rename: dict[str, Any] | None = None
        order: dict[str, Any] | None = None

        if transform == "option_permutation":
            option_order = _nonidentity_option_order(rng)
        else:
            option_order = list(range(len(base.canonical_options)))

        if transform == "entity_rename":
            rename = _rename_map(base.task_family, rng, base.structure)
        elif transform == "order_permutation":
            order = _order_permutation(base.task_family, rng, base.structure)

        prompt = render_prompt(
            base.task_family,
            base.structure,
            base.canonical_options,
            option_order,
            rename,
            order,
        )
        correct_option_index = option_order.index(semantic_correct)
        records.append(
            Record(
                record_id=f"{base.base_id}__{transform}",
                base_id=base.base_id,
                split=base.split,
                task_family=base.task_family,
                transform=transform,
                semantic={
                    "structure": base.structure,
                    "canonical_options": base.canonical_options,
                },
                prompt=prompt,
                option_order=option_order,
                correct_option_index=correct_option_index,
                rename=rename,
                order=order,
            )
        )
    return records
