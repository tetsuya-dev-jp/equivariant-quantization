"""Deterministic procedural data, transformations, and exact answer oracles."""

from .build import SemanticCollisionError, build_dataset
from .oracles import canonical_answer
from .schema import FAMILIES, SPLIT_ORDER, TRANSFORM_AVAILABILITY

__all__ = [
    "FAMILIES",
    "SPLIT_ORDER",
    "TRANSFORM_AVAILABILITY",
    "SemanticCollisionError",
    "build_dataset",
    "canonical_answer",
]
