"""Dataset assembly with deterministic split generation and collision checks."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .generate import build_records, derive_seed, generate_base
from .schema import (
    DEFAULT_SPLIT_SPECS,
    FAMILIES,
    SPLIT_ORDER,
    TRANSFORM_AVAILABILITY,
    BaseInstance,
    Record,
)


class SemanticCollisionError(ValueError):
    """Raised when two generated bases have the same exact semantic key."""


@dataclass(frozen=True)
class Dataset:
    bases: list[BaseInstance]
    records: list[Record]
    manifest: dict[str, Any]


def register_semantic_key(
    seen: dict[str, str],
    semantic_key: str,
    split: str,
) -> None:
    """Reject any exact semantic collision instead of dropping or reusing items."""
    if semantic_key in seen:
        raise SemanticCollisionError(
            "exact semantic collision between split "
            f"{seen[semantic_key]!r} and split {split!r}: {semantic_key}"
        )
    seen[semantic_key] = split


def _resolve_split_specs(config: dict[str, Any]) -> dict[str, dict[str, int]]:
    data_config = config.get("data", {})
    configured = data_config.get("splits", DEFAULT_SPLIT_SPECS)
    specs: dict[str, dict[str, int]] = {}
    for split in SPLIT_ORDER:
        spec = configured.get(split, DEFAULT_SPLIT_SPECS[split])
        count = int(spec["count"])
        seed = int(spec["seed"])
        if count < 0:
            raise ValueError(f"split {split!r} has a negative count")
        specs[split] = {"count": count, "seed": seed}
    return specs


def _resolve_families(config: dict[str, Any]) -> tuple[str, ...]:
    configured = config.get("data", {}).get("families", FAMILIES)
    families = tuple(configured)
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown task families: {unknown}")
    return families


def build_dataset(
    config: dict[str, Any],
    split_names: list[str] | tuple[str, ...] | None = None,
) -> Dataset:
    """Build the requested splits in a fixed order.

    The same semantic key may not appear in two splits or twice in one split.
    Unsupported family/transform pairs have no record at all.
    """
    split_specs = _resolve_split_specs(config)
    families = _resolve_families(config)

    if split_names is None:
        selected_splits = list(SPLIT_ORDER)
    else:
        requested = list(dict.fromkeys(split_names))
        unknown = [split for split in requested if split not in SPLIT_ORDER]
        if unknown:
            raise ValueError(f"unknown splits: {unknown}")
        selected_splits = [split for split in SPLIT_ORDER if split in requested]
    selected_set = set(selected_splits)

    seen: dict[str, str] = {}
    bases: list[BaseInstance] = []
    records: list[Record] = []
    per_split = {
        split: {"bases": 0, "records": 0} for split in selected_splits
    }
    full_per_split = {
        split: {"bases": 0, "records": 0} for split in SPLIT_ORDER
    }
    collision_rejections = {
        split: {family: 0 for family in families} for split in SPLIT_ORDER
    }
    candidate_counts = {
        split: {
            family: {
                "requested": split_specs[split]["count"],
                "tried": 0,
                "rejected": 0,
                "accepted": 0,
                "rejection_reasons": {"duplicate_semantic_key": 0},
            }
            for family in families
        }
        for split in SPLIT_ORDER
    }

    # Always generate every split in fixed order. A requested subset is filtered
    # afterwards so its accepted instances cannot depend on which subset was
    # asked for.
    for split in SPLIT_ORDER:
        spec = split_specs[split]
        structural = split == "structural_holdout"
        for family in families:
            for index in range(spec["count"]):
                base_id = f"{family}-{split}-{index:04d}"
                attempts = 0
                while True:
                    candidate_counts[split][family]["tried"] += 1
                    base = generate_base(
                        task_family=family,
                        base_id=base_id,
                        split=split,
                        seed=derive_seed(spec["seed"], family, index, attempts),
                        structural=structural,
                    )
                    if base.semantic_key not in seen:
                        break
                    collision_rejections[split][family] += 1
                    candidate_counts[split][family]["rejected"] += 1
                    candidate_counts[split][family]["rejection_reasons"][
                        "duplicate_semantic_key"
                    ] += 1
                    attempts += 1
                    if attempts >= 1000:
                        raise SemanticCollisionError(
                            "could not generate a unique semantic instance after "
                            f"1000 attempts for {base_id}"
                        )
                register_semantic_key(seen, base.semantic_key, split)
                candidate_counts[split][family]["accepted"] += 1
                full_per_split[split]["bases"] += 1

                if split not in selected_set:
                    full_per_split[split]["records"] += len(
                        TRANSFORM_AVAILABILITY[family]
                    )
                    continue

                bases.append(base)
                per_split[split]["bases"] += 1
                family_records = build_records(base)
                records.extend(family_records)
                per_split[split]["records"] += len(family_records)
                full_per_split[split]["records"] += len(family_records)

    manifest = {
        "selected_splits": selected_splits,
        "split_specs": {
            split: split_specs[split] for split in selected_splits
        },
        "families": list(families),
        "transform_availability": {
            family: list(TRANSFORM_AVAILABILITY[family]) for family in families
        },
        "counts": {
            "bases": len(bases),
            "records": len(records),
            "per_split": per_split,
        },
        "collision_rejections": {
            split: collision_rejections[split] for split in selected_splits
        },
        "candidate_counts": {
            split: candidate_counts[split] for split in selected_splits
        },
        "full_generation": {
            "order": list(SPLIT_ORDER),
            "counts": {
                "bases": sum(
                    counts["bases"] for counts in full_per_split.values()
                ),
                "records": sum(
                    counts["records"] for counts in full_per_split.values()
                ),
                "per_split": full_per_split,
            },
            "collision_rejections": collision_rejections,
            "candidate_counts": candidate_counts,
        },
        "structural_holdout": {
            "arithmetic_operation_count": [5, 6],
            "boolean_variable_count": 5,
            "graph_node_count": [8, 9, 10],
        },
    }
    return Dataset(bases=bases, records=records, manifest=manifest)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_dataset(dataset: Dataset, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output_path / "bases.jsonl",
        [base.to_dict() for base in dataset.bases],
    )
    _write_jsonl(
        output_path / "records.jsonl",
        [record.to_dict() for record in dataset.records],
    )
    with (output_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset.manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
