"""Calibration pair selection and split validation."""

from dataclasses import dataclass
from typing import Any

CALIBRATION_SPLIT = "calibration"
TRAINED_TRANSFORMS = ("original", "option_permutation")
RECORD_METADATA_FIELDS = (
    "record_id",
    "base_id",
    "split",
    "task_family",
    "transform",
    "prompt",
    "option_order",
)


@dataclass(frozen=True)
class CalibrationPair:
    """One base instance and its original/option-permuted presentations."""

    base_id: str
    original: dict[str, Any]
    option_permutation: dict[str, Any]

    def ordered_records(self) -> list[dict[str, Any]]:
        return [self.original, self.option_permutation]


def record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return the input fields that determine a teacher-cache row."""
    missing = [field for field in RECORD_METADATA_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"record {record.get('record_id')!r} is missing {missing}"
        )
    return {field: record[field] for field in RECORD_METADATA_FIELDS}


def ordered_record_metadata(
    pairs: list[CalibrationPair],
) -> list[dict[str, Any]]:
    return [
        record_metadata(record)
        for pair in pairs
        for record in pair.ordered_records()
    ]


def validate_calibration_records(
    records: list[dict[str, Any]],
    split: str = CALIBRATION_SPLIT,
) -> None:
    if split != CALIBRATION_SPLIT:
        raise ValueError(
            "calibration only supports the literal 'calibration' split, "
            f"got {split!r}"
        )
    if not records:
        raise ValueError("no records supplied")
    wrong = sorted({record.get("split") for record in records} - {split})
    if wrong:
        raise ValueError(
            f"calibration accepts only split {split!r} records; found {wrong}"
        )


def select_calibration_pairs(
    records: list[dict[str, Any]],
    split: str = CALIBRATION_SPLIT,
) -> list[CalibrationPair]:
    """Return exactly one original and one option permutation per base."""
    validate_calibration_records(records, split)

    by_base: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        transform = record.get("transform")
        if transform not in TRAINED_TRANSFORMS:
            continue
        base_id = record["base_id"]
        slots = by_base.setdefault(base_id, {})
        if transform in slots:
            raise ValueError(
                f"base {base_id} has more than one {transform} calibration record"
            )
        slots[transform] = record

    pairs: list[CalibrationPair] = []
    for base_id in sorted(by_base):
        slots = by_base[base_id]
        missing = [name for name in TRAINED_TRANSFORMS if name not in slots]
        if missing:
            raise ValueError(f"base {base_id} is missing calibration records: {missing}")
        original = slots["original"]
        variant = slots["option_permutation"]
        if list(original["option_order"]) == list(variant["option_order"]):
            raise ValueError(
                f"base {base_id} option_permutation is an identity permutation"
            )
        pairs.append(CalibrationPair(base_id, original, variant))

    if not pairs:
        raise ValueError("no original/option_permutation calibration pairs found")
    return pairs
