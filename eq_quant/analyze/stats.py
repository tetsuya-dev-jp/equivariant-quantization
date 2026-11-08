"""Base-cluster paired analysis of canonical predictions.

Pooled estimates are base-macro rates: average within each base problem first,
then average base problems. Raw counts and micro rates are retained separately.
Bootstrap resampling is by base problem and carries every transformation of that
base together.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eq_quant.data.oracles import canonical_answer
from eq_quant.data.schema import TRANSFORMS, available_transforms

TRANSFORM_GROUPS = ("all", *TRANSFORMS)
BOOTSTRAP_METRICS = (
    "delta_accuracy_macro",
    "delta_consistency_macro",
    "broken_consistent_pair_rate_macro",
    "repaired_consistent_pair_rate_macro",
    "answer_flip_rate_macro",
    "union_bound_macro",
)


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    predictions_path = Path(path)
    if predictions_path.is_dir():
        predictions_path = predictions_path / "predictions.jsonl"
    rows: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _condition_sort_key(condition: str) -> tuple[int, str]:
    order = {"fp": 0, "w8": 1, "w4": 2, "w3": 3}
    return (order.get(condition, 99), condition)


def _recompute_correct(entry: dict[str, Any]) -> bool:
    canonical = canonical_answer(
        entry["task_family"],
        entry["semantic"]["structure"],
        entry["semantic"]["canonical_options"],
    )
    recomputed = entry["canonical_prediction"] == canonical
    if bool(entry.get("correct", recomputed)) != recomputed:
        raise ValueError(
            f"prediction row {entry['record_id']} has an incorrect stored label"
        )
    return recomputed


def _prepare(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted(
        {entry["condition"] for entry in predictions},
        key=_condition_sort_key,
    )
    bases: dict[str, dict[str, Any]] = {}
    for entry in predictions:
        base_id = entry["base_id"]
        base = bases.setdefault(
            base_id,
            {
                "split": entry["split"],
                "task_family": entry["task_family"],
                "conditions": {},
            },
        )
        if base["split"] != entry["split"] or base["task_family"] != entry["task_family"]:
            raise ValueError(f"inconsistent metadata for base {base_id}")
        entry = dict(entry)
        entry["correct"] = _recompute_correct(entry)
        base["conditions"].setdefault(entry["condition"], []).append(entry)

    for base in bases.values():
        for condition, entries in base["conditions"].items():
            if not any(entry["transform"] == "original" for entry in entries):
                raise ValueError(
                    f"base {entries[0]['base_id']} lacks an original record "
                    f"for condition {condition}"
                )
            transforms = [entry["transform"] for entry in entries]
            if len(transforms) != len(set(transforms)):
                raise ValueError(
                    f"duplicate transform rows for base {entries[0]['base_id']} "
                    f"and condition {condition}"
                )
    return {"conditions": conditions, "bases": bases}


def _base_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    original_entries = [entry for entry in entries if entry["transform"] == "original"]
    if len(original_entries) != 1:
        raise ValueError("each base/condition must have exactly one original record")
    original = original_entries[0]
    by_transform = {entry["transform"]: entry for entry in entries}
    variant_transforms = [
        transform for transform in by_transform if transform != "original"
    ]

    n_consistent = 0
    n_jointly_correct = 0
    n_consistent_wrong = 0
    transform_stats: dict[str, dict[str, Any]] = {}

    for transform, entry in by_transform.items():
        prediction = entry["canonical_prediction"]
        if transform == "original":
            consistent = None
            joint_correct = None
            consistent_wrong = None
        else:
            consistent = original["canonical_prediction"] == prediction
            joint_correct = bool(original["correct"] and entry["correct"])
            consistent_wrong = bool(consistent and not original["correct"])
            n_consistent += int(consistent)
            n_jointly_correct += int(joint_correct)
            n_consistent_wrong += int(consistent_wrong)
        transform_stats[transform] = {
            "n_records": 1,
            "n_correct": int(entry["correct"]),
            "accuracy": float(entry["correct"]),
            "consistent": consistent,
            "joint_correct": joint_correct,
            "consistent_wrong": consistent_wrong,
            "prediction": prediction,
        }

    n_records = len(entries)
    n_correct = sum(1 for entry in entries if entry["correct"])
    n_pairs = len(variant_transforms)
    return {
        "n_records": n_records,
        "n_correct": n_correct,
        "accuracy": n_correct / n_records,
        "n_pairs": n_pairs,
        "n_consistent": n_consistent,
        "consistency": n_consistent / n_pairs if n_pairs else None,
        "n_jointly_correct": n_jointly_correct,
        "n_consistent_wrong": n_consistent_wrong,
        "transforms": transform_stats,
        "original_prediction": original["canonical_prediction"],
        "original_correct": bool(original["correct"]),
    }


def _base_transitions(
    reference_metrics: dict[str, Any],
    condition_metrics: dict[str, Any],
    transform: str,
) -> dict[str, int]:
    reference_transforms = reference_metrics["transforms"]
    condition_transforms = condition_metrics["transforms"]
    broken = 0
    repaired = 0
    record_flips = 0
    n_pairs = 0
    n_records_compared = 0
    union_bound_sum = 0

    if transform == "all":
        shared = sorted(
            set(reference_transforms).intersection(condition_transforms)
        )
        n_records_compared = len(shared)
        record_flips = sum(
            int(
                reference_transforms[name]["prediction"]
                != condition_transforms[name]["prediction"]
            )
            for name in shared
        )
        for name in shared:
            if name == "original":
                continue
            reference_pair = reference_transforms[name]
            condition_pair = condition_transforms[name]
            if (
                reference_pair["consistent"] is None
                or condition_pair["consistent"] is None
            ):
                continue
            n_pairs += 1
            if reference_pair["consistent"] and not condition_pair["consistent"]:
                broken += 1
            if not reference_pair["consistent"] and condition_pair["consistent"]:
                repaired += 1
            flip_original = int(
                reference_transforms["original"]["prediction"]
                != condition_transforms["original"]["prediction"]
            )
            flip_variant = int(
                reference_pair["prediction"] != condition_pair["prediction"]
            )
            union_bound = flip_original + flip_variant
            correspondence_change = abs(
                int(condition_pair["consistent"])
                - int(reference_pair["consistent"])
            )
            if correspondence_change > union_bound:
                raise ValueError("union-bound diagnostic violated for one pair")
            union_bound_sum += union_bound
    elif transform == "original":
        if (
            "original" in reference_transforms
            and "original" in condition_transforms
        ):
            n_records_compared = 1
            record_flips = int(
                reference_transforms["original"]["prediction"]
                != condition_transforms["original"]["prediction"]
            )
    else:
        if (
            transform in reference_transforms
            and transform in condition_transforms
            and "original" in reference_transforms
            and "original" in condition_transforms
        ):
            reference_pair = reference_transforms[transform]
            condition_pair = condition_transforms[transform]
            n_pairs = 1
            n_records_compared = 2
            if reference_pair["consistent"] and not condition_pair["consistent"]:
                broken = 1
            if not reference_pair["consistent"] and condition_pair["consistent"]:
                repaired = 1
            flip_original = int(
                reference_transforms["original"]["prediction"]
                != condition_transforms["original"]["prediction"]
            )
            flip_variant = int(
                reference_pair["prediction"] != condition_pair["prediction"]
            )
            record_flips = flip_original + flip_variant
            union_bound_sum = record_flips
            correspondence_change = abs(
                int(condition_pair["consistent"])
                - int(reference_pair["consistent"])
            )
            if correspondence_change > union_bound_sum:
                raise ValueError("union-bound diagnostic violated for one pair")

    return {
        "n_pairs": n_pairs,
        "n_records_compared": n_records_compared,
        "broken": broken,
        "repaired": repaired,
        "record_flips": record_flips,
        "union_bound_sum": union_bound_sum,
    }


def _base_summary(
    base: dict[str, Any],
    conditions: list[str],
    reference: str,
) -> dict[str, Any]:
    metrics = {
        condition: _base_metrics(base["conditions"][condition])
        for condition in conditions
        if condition in base["conditions"]
    }
    transitions: dict[str, dict[str, dict[str, int]]] = {}
    if reference in metrics:
        for condition in conditions:
            if condition == reference or condition not in metrics:
                continue
            transitions[condition] = {
                transform: _base_transitions(
                    metrics[reference], metrics[condition], transform
                )
                for transform in TRANSFORM_GROUPS
            }
    return {
        "split": base["split"],
        "task_family": base["task_family"],
        "metrics": metrics,
        "transitions": transitions,
    }


def _mean(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _aggregate_condition(
    base_summaries: list[dict[str, Any]],
    condition: str,
    transform: str,
) -> dict[str, Any]:
    accuracies: list[float] = []
    consistencies: list[float] = []
    n_bases = 0
    n_records = 0
    n_correct = 0
    n_pairs = 0
    n_consistent = 0
    jointly_correct = 0
    consistent_wrong = 0

    for summary in base_summaries:
        metrics = summary["metrics"].get(condition)
        if metrics is None:
            continue
        if transform == "all":
            accuracy = metrics["accuracy"]
            consistency_value = (
                float(metrics["consistency"])
                if metrics["consistency"] is not None
                else None
            )
            n_records += metrics["n_records"]
            n_correct += metrics["n_correct"]
            n_pairs += metrics["n_pairs"]
            n_consistent += metrics["n_consistent"]
            jointly_correct += metrics["n_jointly_correct"]
            consistent_wrong += metrics["n_consistent_wrong"]
        else:
            transform_metrics = metrics["transforms"].get(transform)
            if transform_metrics is None:
                continue
            accuracy = transform_metrics["accuracy"]
            consistency_value = (
                None
                if transform_metrics["consistent"] is None
                else float(bool(transform_metrics["consistent"]))
            )
            n_records += 1
            n_correct += transform_metrics["n_correct"]
            if transform != "original":
                n_pairs += 1
                n_consistent += int(bool(transform_metrics["consistent"]))
                jointly_correct += int(bool(transform_metrics["joint_correct"]))
                consistent_wrong += int(bool(transform_metrics["consistent_wrong"]))
        n_bases += 1
        accuracies.append(float(accuracy))
        if consistency_value is not None:
            consistencies.append(consistency_value)

    return {
        "n_bases": n_bases,
        "n_records": n_records,
        "n_correct": n_correct,
        "accuracy_macro": _mean(accuracies),
        "accuracy_micro": n_correct / n_records if n_records else None,
        "n_pairs": n_pairs,
        "n_consistent": n_consistent,
        "consistency_macro": _mean(consistencies),
        "consistency_micro": n_consistent / n_pairs if n_pairs else None,
        "jointly_correct_pairs": jointly_correct,
        "consistent_but_wrong_pairs": consistent_wrong,
    }


def _empty_condition_aggregate() -> dict[str, Any]:
    return {
        "n_bases": 0,
        "n_records": 0,
        "n_correct": 0,
        "accuracy_macro": None,
        "accuracy_micro": None,
        "n_pairs": 0,
        "n_consistent": 0,
        "consistency_macro": None,
        "consistency_micro": None,
        "jointly_correct_pairs": 0,
        "consistent_but_wrong_pairs": 0,
    }


def _aggregate_comparison(
    base_summaries: list[dict[str, Any]],
    condition: str,
    transform: str,
    reference: str,
) -> dict[str, Any]:
    delta_accuracies: list[float] = []
    delta_consistencies: list[float] = []
    broken = 0
    repaired = 0
    answer_flips = 0
    broken_rates: list[float] = []
    repaired_rates: list[float] = []
    flip_rates: list[float] = []
    union_bounds: list[float] = []
    n_bases = 0

    for summary in base_summaries:
        reference_metrics = summary["metrics"].get(reference)
        condition_metrics = summary["metrics"].get(condition)
        if reference_metrics is None or condition_metrics is None:
            continue
        if transform == "all":
            reference_accuracy = reference_metrics["accuracy"]
            condition_accuracy = condition_metrics["accuracy"]
            reference_consistency = reference_metrics["consistency"]
            condition_consistency = condition_metrics["consistency"]
            reference_consistency_value = (
                float(reference_consistency)
                if reference_consistency is not None
                else None
            )
            condition_consistency_value = (
                float(condition_consistency)
                if condition_consistency is not None
                else None
            )
        else:
            reference_transform = reference_metrics["transforms"].get(transform)
            condition_transform = condition_metrics["transforms"].get(transform)
            if reference_transform is None or condition_transform is None:
                continue
            reference_accuracy = reference_transform["accuracy"]
            condition_accuracy = condition_transform["accuracy"]
            reference_consistency = reference_transform["consistent"]
            condition_consistency = condition_transform["consistent"]
            reference_consistency_value = (
                None
                if reference_consistency is None
                else float(bool(reference_consistency))
            )
            condition_consistency_value = (
                None
                if condition_consistency is None
                else float(bool(condition_consistency))
            )

        n_bases += 1
        delta_accuracies.append(
            float(condition_accuracy) - float(reference_accuracy)
        )
        if (
            reference_consistency_value is not None
            and condition_consistency_value is not None
        ):
            delta_consistencies.append(
                condition_consistency_value - reference_consistency_value
            )

        transition = summary["transitions"][condition][transform]
        broken += transition["broken"]
        repaired += transition["repaired"]
        answer_flips += transition["record_flips"]
        if transition["n_pairs"] > 0:
            broken_rates.append(transition["broken"] / transition["n_pairs"])
            repaired_rates.append(
                transition["repaired"] / transition["n_pairs"]
            )
            union_bounds.append(
                transition["union_bound_sum"] / transition["n_pairs"]
            )
        if transition["n_records_compared"] > 0:
            flip_rates.append(
                transition["record_flips"] / transition["n_records_compared"]
            )

    condition_aggregate = _aggregate_condition(
        base_summaries, condition, transform
    )
    delta_accuracy_macro = _mean(delta_accuracies)
    delta_consistency_macro = _mean(delta_consistencies)
    union_bound_macro = _mean(union_bounds)
    abs_delta_consistency = (
        abs(delta_consistency_macro)
        if delta_consistency_macro is not None
        else None
    )
    if abs_delta_consistency is not None and union_bound_macro is not None:
        union_bound_satisfied: bool | None = (
            abs_delta_consistency <= union_bound_macro + 1e-12
        )
    else:
        union_bound_satisfied = None

    return {
        "n_bases": n_bases,
        "delta_accuracy_macro": delta_accuracy_macro,
        "delta_consistency_macro": delta_consistency_macro,
        "accuracy_macro": condition_aggregate["accuracy_macro"],
        "consistency_macro": condition_aggregate["consistency_macro"],
        "broken_consistent_pairs": broken,
        "broken_consistent_pair_rate_macro": _mean(broken_rates),
        "repaired_consistent_pairs": repaired,
        "repaired_consistent_pair_rate_macro": _mean(repaired_rates),
        "answer_flips": answer_flips,
        "answer_flip_rate_macro": _mean(flip_rates),
        "union_bound_macro": union_bound_macro,
        "abs_delta_consistency_macro": abs_delta_consistency,
        "union_bound_satisfied": union_bound_satisfied,
        "jointly_correct_pairs": condition_aggregate["jointly_correct_pairs"],
        "consistent_but_wrong_pairs": condition_aggregate[
            "consistent_but_wrong_pairs"
        ],
    }


def _empty_comparison() -> dict[str, Any]:
    return {
        "n_bases": 0,
        "delta_accuracy_macro": None,
        "delta_consistency_macro": None,
        "accuracy_macro": None,
        "consistency_macro": None,
        "broken_consistent_pairs": 0,
        "broken_consistent_pair_rate_macro": None,
        "repaired_consistent_pairs": 0,
        "repaired_consistent_pair_rate_macro": None,
        "answer_flips": 0,
        "answer_flip_rate_macro": None,
        "union_bound_macro": None,
        "abs_delta_consistency_macro": None,
        "union_bound_satisfied": None,
        "jointly_correct_pairs": 0,
        "consistent_but_wrong_pairs": 0,
    }


def _group_summaries(
    bases: dict[str, dict[str, Any]],
    conditions: list[str],
    reference: str,
) -> list[tuple[tuple[str, str], list[dict[str, Any]]]]:
    summaries = [
        _base_summary(base, conditions, reference) for base in bases.values()
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    groups[("all", "all")] = summaries
    for split in sorted({summary["split"] for summary in summaries}):
        groups[(split, "all")] = [
            summary for summary in summaries if summary["split"] == split
        ]
    for family in sorted({summary["task_family"] for summary in summaries}):
        groups[("all", family)] = [
            summary for summary in summaries if summary["task_family"] == family
        ]
        for split in sorted({summary["split"] for summary in summaries}):
            selected = [
                summary
                for summary in summaries
                if summary["split"] == split
                and summary["task_family"] == family
            ]
            if selected:
                groups[(split, family)] = selected
    return sorted(groups.items())


def _availability_for_group(base_summaries: list[dict[str, Any]]) -> set[str]:
    available: set[str] = set()
    for summary in base_summaries:
        available.update(
            available_transforms(summary["split"], summary["task_family"])
        )
    return available


def _bootstrap(
    base_summaries: list[dict[str, Any]],
    conditions: list[str],
    transforms: list[str],
    n_boot: int,
    seed: int,
    reference: str,
) -> dict[tuple[str, str], dict[str, list[float]]]:
    compared_conditions = [
        condition for condition in conditions if condition != reference
    ]
    if (
        reference not in conditions
        or not compared_conditions
        or not base_summaries
    ):
        return {}

    rng = np.random.default_rng(seed)
    n_bases = len(base_summaries)
    samples: dict[tuple[str, str], dict[str, list[float]]] = {
        (condition, transform): {metric: [] for metric in BOOTSTRAP_METRICS}
        for condition in compared_conditions
        for transform in transforms
    }
    for _ in range(n_boot):
        indices = rng.integers(0, n_bases, size=n_bases)
        resampled = [base_summaries[index] for index in indices]
        for condition in compared_conditions:
            for transform in transforms:
                values = _aggregate_comparison(
                    resampled, condition, transform, reference
                )
                for metric in BOOTSTRAP_METRICS:
                    value = values[metric]
                    if value is None:
                        continue
                    if isinstance(value, float) and math.isnan(value):
                        continue
                    samples[(condition, transform)][metric].append(float(value))
    return samples


def _interval(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"ci_low": None, "ci_high": None}
    low, high = np.percentile(values, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high)}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def analyze_predictions(
    predictions_path: str | Path,
    output_dir: str | Path,
    n_boot: int = 1000,
    seed: int = 4108,
    conditions: list[str] | None = None,
    bootstrap_seed: int | None = None,
    bootstrap_samples: int | None = None,
    reference: str = "fp",
) -> dict[str, Any]:
    predictions = load_predictions(predictions_path)
    if not predictions:
        raise ValueError("no predictions to analyze")
    if bootstrap_seed is not None:
        seed = bootstrap_seed
    if bootstrap_samples is not None:
        n_boot = bootstrap_samples

    prepared = _prepare(predictions)
    selected_conditions = prepared["conditions"]
    if conditions is not None:
        selected_conditions = [
            condition for condition in selected_conditions if condition in conditions
        ]
    if not selected_conditions:
        raise ValueError("no selected conditions are present in the predictions")
    if reference not in selected_conditions:
        raise ValueError(f"analysis requires the {reference} condition")

    grouped = _group_summaries(
        prepared["bases"], selected_conditions, reference
    )
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []

    for (split, family), base_summaries in grouped:
        availability = _availability_for_group(base_summaries)
        available_transforms = [
            transform
            for transform in TRANSFORM_GROUPS
            if transform == "all" or transform in availability
        ]

        for condition in selected_conditions:
            for transform in TRANSFORM_GROUPS:
                available = transform in available_transforms
                aggregate = (
                    _aggregate_condition(base_summaries, condition, transform)
                    if available
                    else _empty_condition_aggregate()
                )
                summary_rows.append(
                    {
                        "split": split,
                        "task_family": family,
                        "condition": condition,
                        "transform": transform,
                        "available": available,
                        **aggregate,
                    }
                )

        for condition in selected_conditions:
            if condition == reference:
                continue
            for transform in TRANSFORM_GROUPS:
                available = transform in available_transforms
                comparison = (
                    _aggregate_comparison(
                        base_summaries, condition, transform, reference
                    )
                    if available
                    else _empty_comparison()
                )
                comparison_rows.append(
                    {
                        "split": split,
                        "task_family": family,
                        "condition": condition,
                        "transform": transform,
                        "reference": reference,
                        "available": available,
                        **comparison,
                    }
                )

        bootstrap = _bootstrap(
            base_summaries,
            selected_conditions,
            available_transforms,
            n_boot=n_boot,
            seed=seed,
            reference=reference,
        )
        for (condition, transform), metrics in bootstrap.items():
            point = _aggregate_comparison(
                base_summaries, condition, transform, reference
            )
            for metric, values in metrics.items():
                if not values:
                    continue
                interval = _interval(values)
                bootstrap_rows.append(
                    {
                        "split": split,
                        "task_family": family,
                        "condition": condition,
                        "transform": transform,
                        "reference": reference,
                        "metric": metric,
                        "estimate": point.get(metric),
                        "ci_low": interval["ci_low"],
                        "ci_high": interval["ci_high"],
                        "n_boot": len(values),
                        "seed": seed,
                    }
                )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {
        "predictions": str(predictions_path),
        "conditions": selected_conditions,
        "bootstrap": {
            "unit": "base_id",
            "samples": n_boot,
            "seed": seed,
            "estimator": "base_macro",
        },
        "summary": summary_rows,
        "comparisons": comparison_rows,
        "bootstrap_intervals": bootstrap_rows,
    }
    with (output_path / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    _write_csv(
        output_path / "summary.csv",
        summary_rows,
        [
            "split",
            "task_family",
            "condition",
            "transform",
            "available",
            "n_bases",
            "n_records",
            "n_correct",
            "accuracy_macro",
            "accuracy_micro",
            "n_pairs",
            "n_consistent",
            "consistency_macro",
            "consistency_micro",
            "jointly_correct_pairs",
            "consistent_but_wrong_pairs",
        ],
    )
    _write_csv(
        output_path / "comparisons.csv",
        comparison_rows,
        [
            "split",
            "task_family",
            "condition",
            "transform",
            "reference",
            "available",
            "n_bases",
            "delta_accuracy_macro",
            "delta_consistency_macro",
            "accuracy_macro",
            "consistency_macro",
            "broken_consistent_pairs",
            "broken_consistent_pair_rate_macro",
            "repaired_consistent_pairs",
            "repaired_consistent_pair_rate_macro",
            "answer_flips",
            "answer_flip_rate_macro",
            "union_bound_macro",
            "abs_delta_consistency_macro",
            "union_bound_satisfied",
            "jointly_correct_pairs",
            "consistent_but_wrong_pairs",
        ],
    )
    _write_csv(
        output_path / "bootstrap.csv",
        bootstrap_rows,
        [
            "split",
            "task_family",
            "condition",
            "transform",
            "reference",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "n_boot",
            "seed",
        ],
    )
    return result
