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

BASE_TRANSFORMS = (
    "original",
    "option_permutation",
    "entity_rename",
    "order_permutation",
)

COMPOSED_TRANSFORMS = (
    "option_permutation+entity_rename",
    "option_permutation+entity_rename+order_permutation",
)

# Analysis enumerates every defined transform, including compositions.
TRANSFORMS = (*BASE_TRANSFORMS, *COMPOSED_TRANSFORMS)

# Unsupported family/transform pairs are absent, never represented as identity.
BASE_TRANSFORM_AVAILABILITY = {
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

# Compositions are evaluation-only and gated to the final evaluation splits.
COMPOSED_SPLITS = ("final", "structural_holdout")

COMPOSED_TRANSFORM_AVAILABILITY = {
    "arithmetic": ("option_permutation+entity_rename",),
    "boolean": COMPOSED_TRANSFORMS,
    "graph": COMPOSED_TRANSFORMS,
}

# Family-level union used by analysis enumeration; generation applies the
# split gate through available_transforms.
TRANSFORM_AVAILABILITY = {
    family: (
        *BASE_TRANSFORM_AVAILABILITY[family],
        *COMPOSED_TRANSFORM_AVAILABILITY[family],
    )
    for family in FAMILIES
}


def available_transforms(split: str, family: str) -> tuple[str, ...]:
    """Return the transforms generated for one split and family."""
    if split in COMPOSED_SPLITS:
        return (
            *BASE_TRANSFORM_AVAILABILITY[family],
            *COMPOSED_TRANSFORM_AVAILABILITY[family],
        )
    return BASE_TRANSFORM_AVAILABILITY[family]

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
