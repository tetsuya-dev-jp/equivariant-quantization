"""Shared constants and lightweight record containers for the data layer."""

from dataclasses import asdict, dataclass
from typing import Any

SPLIT_ORDER = (
    "calibration",
    "development",
    "pilot",
    "final",
    "structural_holdout",
)

FAMILIES = ("arithmetic", "boolean", "graph")

TRANSFORMS = (
    "original",
    "option_permutation",
    "entity_rename",
    "order_permutation",
)

# Unsupported family/transform pairs are absent, never represented as identity.
TRANSFORM_AVAILABILITY = {
    "arithmetic": ("original", "option_permutation", "entity_rename"),
    "boolean": (
        "original",
        "option_permutation",
        "entity_rename",
        "order_permutation",
    ),
    "graph": (
        "original",
        "option_permutation",
        "entity_rename",
        "order_permutation",
    ),
}

OPTION_LABELS = ("A", "B", "C", "D")

DEFAULT_SPLIT_SPECS = {
    "calibration": {"count": 128, "seed": 1108},
    "development": {"count": 128, "seed": 2108},
    "pilot": {"count": 64, "seed": 3108},
    "final": {"count": 256, "seed": 4108},
    "structural_holdout": {"count": 128, "seed": 5108},
}


@dataclass(frozen=True)
class BaseInstance:
    """A canonical semantic question before any presentation transform."""

    base_id: str
    split: str
    task_family: str
    structure: dict[str, Any]
    canonical_options: list[Any]
    semantic_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Record:
    """One rendered presentation of a base instance."""

    record_id: str
    base_id: str
    split: str
    task_family: str
    transform: str
    semantic: dict[str, Any]
    prompt: str
    option_order: list[int]
    correct_option_index: int
    rename: dict[str, Any] | None
    order: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
