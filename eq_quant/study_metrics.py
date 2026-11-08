"""Base-clustered metrics for the frozen calibration study.

Prediction rows follow the evaluator format. Correctness is always recomputed
from ``eq_quant.data.oracles.canonical_answer``; a saved ``correct`` label is
only checked for consistency, never trusted.

Public functions operate on lists of prediction row dictionaries and return
plain dictionaries. The main entry points are:

``group_by_condition_and_base``
    Validate rows and group the exact transforms available per condition/base.

``summarize_conditions`` / ``summarize_condition``
    Per-condition, base-macro summaries with endpoint subsets.

``compare_conditions``
    Paired base-cluster differences with percentile bootstrap CIs.

``select_lambda``
    Frozen development selection rule for regularized objectives.

Endpoint names are ``all_variants`` (every non-original transform),
``held_out_singles`` (``entity_rename`` and ``order_permutation``), and
``compositions`` (transforms containing ``+``). An endpoint without any
available transform in a base is represented by ``None``.

All population estimates are base-macro: every base receives equal weight,
regardless of how many transforms are available. Counts are reported next to
the macro rates. Bootstrap resampling is by base, carrying all of that base's
transforms together.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from eq_quant.data.oracles import canonical_answer

ORIGINAL_TRANSFORM = "original"
HELD_OUT_TRANSFORMS = ("entity_rename", "order_permutation")
ENDPOINT_NAMES = ("all_variants", "held_out_singles", "compositions")

BASE_METRIC_KEYS = (
    "accuracy",
    "original_accuracy",
    "transformed_accuracy",
    "consistency",
    "jointly_correct_rate",
    "consistent_wrong_rate",
    "held_out_accuracy",
    "held_out_consistency",
    "held_out_jointly_correct_rate",
    "held_out_consistent_wrong_rate",
    "composition_accuracy",
    "composition_consistency",
    "composition_jointly_correct_rate",
    "composition_consistent_wrong_rate",
)

FIDELITY_METRIC_KEYS = (
    "teacher_fidelity",
    "held_out_teacher_fidelity",
    "composition_teacher_fidelity",
)

SELECTION_KEYS = (
    "lambda",
    "overall_accuracy",
    "held_out_jointly_correct_rate",
    "held_out_consistency",
)


def recompute_correct(row: Mapping[str, Any]) -> bool:
    """Return correctness recomputed from the saved semantic structure.

    The stored ``correct`` label, when present, must agree with the recomputed
    value. A saved ``prediction_display``/``option_order`` mapping and
    ``correct_option_index`` are also checked when present, which catches
    non-self-inverse option mappings that would otherwise silently mislabel a
    row.
    """
    canonical_options = row["semantic"]["canonical_options"]
    gold = canonical_answer(
        row["task_family"],
        row["semantic"]["structure"],
        canonical_options,
    )
    prediction = int(row["canonical_prediction"])
    correct = prediction == gold

    if "correct" in row and bool(row["correct"]) != correct:
        raise ValueError(
            f"prediction row {row.get('record_id', row.get('base_id'))!r} "
            "has an incorrect stored correct label"
        )

    option_order = row.get("option_order")
    if option_order is not None:
        display_to_canonical = [int(index) for index in option_order]
        if sorted(display_to_canonical) != list(range(len(canonical_options))):
            raise ValueError(
                f"row {row.get('record_id', row.get('base_id'))!r} has an "
                "option_order that is not a permutation of the canonical options"
            )
        if "correct_option_index" in row:
            expected_display = display_to_canonical.index(gold)
            if int(row["correct_option_index"]) != expected_display:
                raise ValueError(
                    f"row {row.get('record_id', row.get('base_id'))!r} has an "
                    "incorrect saved correct_option_index"
                )
        if "prediction_display" in row:
            display_prediction = int(row["prediction_display"])
            if display_to_canonical[display_prediction] != prediction:
                raise ValueError(
                    f"row {row.get('record_id', row.get('base_id'))!r} has a "
                    "canonical_prediction inconsistent with option_order"
                )

    return correct


def group_by_condition_and_base(
    predictions: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Validate predictions and group rows by condition and base.

    Returns ``grouped[condition][base_id]`` with keys ``base_id``, ``split``,
    ``task_family``, ``semantic``, and ``rows`` (transform -> enriched row).
    Each enriched row carries ``_recomputed_correct``.

    Validation requires consistent base metadata across every condition, no
    duplicate transform within a condition/base, exactly one original row, and
    at least one row overall.
    """
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    base_metadata: dict[str, dict[str, Any]] = {}
    seen_rows = False

    for raw_row in predictions:
        seen_rows = True
        row = dict(raw_row)
        for key in ("condition", "base_id", "transform", "split", "task_family"):
            if key not in row:
                raise ValueError(f"prediction row is missing required key {key!r}")
        if not str(row["condition"]).strip():
            raise ValueError("prediction row has an empty condition")
        if not str(row["base_id"]).strip():
            raise ValueError("prediction row has an empty base_id")

        condition = str(row["condition"])
        base_id = str(row["base_id"])
        transform = str(row["transform"])
        metadata = {
            "split": row["split"],
            "task_family": row["task_family"],
            "semantic": row["semantic"],
        }
        existing_metadata = base_metadata.setdefault(base_id, metadata)
        if existing_metadata != metadata:
            raise ValueError(f"inconsistent metadata for base {base_id!r}")

        row["_recomputed_correct"] = recompute_correct(row)
        base_entry = grouped.setdefault(condition, {}).setdefault(
            base_id,
            {
                "base_id": base_id,
                "split": row["split"],
                "task_family": row["task_family"],
                "semantic": row["semantic"],
                "rows": {},
            },
        )
        if transform in base_entry["rows"]:
            raise ValueError(
                f"duplicate transform row for base {base_id!r} "
                f"and condition {condition!r}: {transform!r}"
            )
        base_entry["rows"][transform] = row

    if not seen_rows:
        raise ValueError("no prediction rows supplied")

    for condition, bases in grouped.items():
        for base_id, entry in bases.items():
            if ORIGINAL_TRANSFORM not in entry["rows"]:
                raise ValueError(
                    f"base {base_id!r} lacks an original record "
                    f"for condition {condition!r}"
                )
    return grouped


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _endpoint_transforms(transforms: Iterable[str]) -> dict[str, list[str]]:
    names = list(transforms)
    return {
        "all_variants": names,
        "held_out_singles": [
            transform for transform in names if transform in HELD_OUT_TRANSFORMS
        ],
        "compositions": [
            transform for transform in names if "+" in transform
        ],
    }


def _variant_stats(
    original_row: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    transforms: Sequence[str],
) -> dict[str, Any] | None:
    if not transforms:
        return None

    original_prediction = original_row["canonical_prediction"]
    original_correct = bool(original_row["_recomputed_correct"])
    n_correct = 0
    n_consistent = 0
    n_jointly_correct = 0
    n_consistent_wrong = 0

    for transform in transforms:
        row = rows[transform]
        prediction = row["canonical_prediction"]
        correct = bool(row["_recomputed_correct"])
        consistent = prediction == original_prediction
        n_correct += int(correct)
        n_consistent += int(consistent)
        n_jointly_correct += int(original_correct and correct)
        n_consistent_wrong += int(consistent and not original_correct)

    n_pairs = len(transforms)
    return {
        "n_records": n_pairs,
        "n_correct": n_correct,
        "accuracy": _ratio(n_correct, n_pairs),
        "n_pairs": n_pairs,
        "n_consistent": n_consistent,
        "consistency": _ratio(n_consistent, n_pairs),
        "n_jointly_correct": n_jointly_correct,
        "jointly_correct_rate": _ratio(n_jointly_correct, n_pairs),
        "n_consistent_wrong": n_consistent_wrong,
        "consistent_wrong_rate": _ratio(n_consistent_wrong, n_pairs),
    }


def _base_metrics(entry: Mapping[str, Any]) -> dict[str, Any]:
    rows = entry["rows"]
    original_row = rows[ORIGINAL_TRANSFORM]
    variant_names = [
        transform for transform in rows if transform != ORIGINAL_TRANSFORM
    ]
    endpoint_transforms = _endpoint_transforms(variant_names)
    endpoints = {
        name: _variant_stats(original_row, rows, endpoint_transforms[name])
        for name in ENDPOINT_NAMES
    }

    n_records = len(rows)
    n_correct = sum(
        int(row["_recomputed_correct"]) for row in rows.values()
    )
    return {
        "base_id": entry["base_id"],
        "split": entry["split"],
        "task_family": entry["task_family"],
        "n_records": n_records,
        "n_correct": n_correct,
        "accuracy": _ratio(n_correct, n_records),
        "original_accuracy": float(original_row["_recomputed_correct"]),
        "original_correct": bool(original_row["_recomputed_correct"]),
        "original_prediction": original_row["canonical_prediction"],
        "transformed": endpoints["all_variants"],
        "endpoints": endpoints,
    }


def _aggregate_endpoint(
    base_stats: Sequence[Mapping[str, Any]],
    endpoint: str,
) -> dict[str, Any] | None:
    stats = [
        base["endpoints"][endpoint]
        for base in base_stats
        if base["endpoints"][endpoint] is not None
    ]
    if not stats:
        return None

    n_records = sum(value["n_records"] for value in stats)
    n_correct = sum(value["n_correct"] for value in stats)
    n_pairs = sum(value["n_pairs"] for value in stats)
    n_consistent = sum(value["n_consistent"] for value in stats)
    n_jointly_correct = sum(value["n_jointly_correct"] for value in stats)
    n_consistent_wrong = sum(value["n_consistent_wrong"] for value in stats)
    return {
        "n_bases": len(stats),
        "n_records": n_records,
        "n_correct": n_correct,
        "accuracy_macro": _mean([value["accuracy"] for value in stats]),
        "accuracy_micro": _ratio(n_correct, n_records),
        "n_pairs": n_pairs,
        "n_consistent": n_consistent,
        "consistency_macro": _mean(
            [value["consistency"] for value in stats]
        ),
        "consistency_micro": _ratio(n_consistent, n_pairs),
        "n_jointly_correct": n_jointly_correct,
        "jointly_correct_rate_macro": _mean(
            [value["jointly_correct_rate"] for value in stats]
        ),
        "jointly_correct_rate_micro": _ratio(n_jointly_correct, n_pairs),
        "n_consistent_wrong": n_consistent_wrong,
        "consistent_wrong_rate_macro": _mean(
            [value["consistent_wrong_rate"] for value in stats]
        ),
        "consistent_wrong_rate_micro": _ratio(n_consistent_wrong, n_pairs),
    }


def _summarize_base_stats(
    condition: str,
    base_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    n_records = sum(base["n_records"] for base in base_stats)
    n_correct = sum(base["n_correct"] for base in base_stats)
    n_original_correct = sum(
        int(base["original_correct"]) for base in base_stats
    )
    endpoints = {
        endpoint: _aggregate_endpoint(base_stats, endpoint)
        for endpoint in ENDPOINT_NAMES
    }
    transformed = endpoints["all_variants"]
    return {
        "condition": condition,
        "n_bases": len(base_stats),
        "n_records": n_records,
        "n_correct": n_correct,
        "accuracy_macro": _mean([base["accuracy"] for base in base_stats]),
        "accuracy_micro": _ratio(n_correct, n_records),
        "n_original_bases": len(base_stats),
        "n_original_correct": n_original_correct,
        "original_accuracy_macro": _mean(
            [base["original_accuracy"] for base in base_stats]
        ),
        "original_accuracy_micro": _ratio(
            n_original_correct, len(base_stats)
        ),
        "transformed": transformed,
        "consistency_macro": (
            transformed["consistency_macro"]
            if transformed is not None
            else None
        ),
        "jointly_correct_rate_macro": (
            transformed["jointly_correct_rate_macro"]
            if transformed is not None
            else None
        ),
        "consistent_wrong_rate_macro": (
            transformed["consistent_wrong_rate_macro"]
            if transformed is not None
            else None
        ),
        "endpoints": endpoints,
        "bases": {
            base["base_id"]: base for base in base_stats
        },
    }


def summarize_conditions(
    predictions: Iterable[Mapping[str, Any]],
    conditions: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return per-condition base-macro summaries for the requested conditions.

    Keys of each summary include ``n_bases``, record/correct counts, macro and
    micro accuracy, original accuracy, transformed-variant metrics,
    ``endpoints`` (with ``None`` for unavailable subsets), and per-base results.
    """
    grouped = group_by_condition_and_base(predictions)
    selected = sorted(grouped) if conditions is None else list(conditions)
    missing = [condition for condition in selected if condition not in grouped]
    if missing:
        raise ValueError(f"conditions are absent from the predictions: {missing}")

    summaries: dict[str, dict[str, Any]] = {}
    for condition in selected:
        base_stats = [
            _base_metrics(entry)
            for entry in grouped[condition].values()
        ]
        summaries[condition] = _summarize_base_stats(condition, base_stats)
    return summaries


def summarize_condition(
    predictions: Iterable[Mapping[str, Any]],
    condition: str,
) -> dict[str, Any]:
    """Return the base-macro summary for one condition."""
    return summarize_conditions(predictions, [condition])[condition]


def _endpoint_metric(
    base: Mapping[str, Any],
    endpoint: str,
    field: str,
) -> float | None:
    endpoint_stats = base["endpoints"][endpoint]
    if endpoint_stats is None:
        return None
    return endpoint_stats[field]


def _base_comparison_values(base: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "accuracy": base["accuracy"],
        "original_accuracy": base["original_accuracy"],
        "transformed_accuracy": _endpoint_metric(
            base, "all_variants", "accuracy"
        ),
        "consistency": _endpoint_metric(
            base, "all_variants", "consistency"
        ),
        "jointly_correct_rate": _endpoint_metric(
            base, "all_variants", "jointly_correct_rate"
        ),
        "consistent_wrong_rate": _endpoint_metric(
            base, "all_variants", "consistent_wrong_rate"
        ),
        "held_out_accuracy": _endpoint_metric(
            base, "held_out_singles", "accuracy"
        ),
        "held_out_consistency": _endpoint_metric(
            base, "held_out_singles", "consistency"
        ),
        "held_out_jointly_correct_rate": _endpoint_metric(
            base, "held_out_singles", "jointly_correct_rate"
        ),
        "held_out_consistent_wrong_rate": _endpoint_metric(
            base, "held_out_singles", "consistent_wrong_rate"
        ),
        "composition_accuracy": _endpoint_metric(
            base, "compositions", "accuracy"
        ),
        "composition_consistency": _endpoint_metric(
            base, "compositions", "consistency"
        ),
        "composition_jointly_correct_rate": _endpoint_metric(
            base, "compositions", "jointly_correct_rate"
        ),
        "composition_consistent_wrong_rate": _endpoint_metric(
            base, "compositions", "consistent_wrong_rate"
        ),
    }


def _fidelity_rate(
    condition_rows: Mapping[str, Mapping[str, Any]],
    reference_rows: Mapping[str, Mapping[str, Any]],
    transforms: Sequence[str],
) -> float | None:
    names = [
        transform for transform in transforms if transform in reference_rows
    ]
    if not names:
        return None
    matches = sum(
        int(
            condition_rows[transform]["canonical_prediction"]
            == reference_rows[transform]["canonical_prediction"]
        )
        for transform in names
    )
    return float(matches / len(names))


def _fidelity_values(
    condition_rows: Mapping[str, Mapping[str, Any]],
    reference_rows: Mapping[str, Mapping[str, Any]] | None,
    common_transforms: Sequence[str],
) -> dict[str, float | None]:
    if reference_rows is None:
        return {key: None for key in FIDELITY_METRIC_KEYS}
    return {
        "teacher_fidelity": _fidelity_rate(
            condition_rows, reference_rows, common_transforms
        ),
        "held_out_teacher_fidelity": _fidelity_rate(
            condition_rows,
            reference_rows,
            [
                transform
                for transform in common_transforms
                if transform in HELD_OUT_TRANSFORMS
            ],
        ),
        "composition_teacher_fidelity": _fidelity_rate(
            condition_rows,
            reference_rows,
            [
                transform
                for transform in common_transforms
                if "+" in transform
            ],
        ),
    }


def _point_means(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    means = np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=float),
        where=counts > 0,
    )
    return means, counts


def _bootstrap_intervals(
    values: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_bases = values.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_bases, size=(n_boot, n_bases))
    samples = values[indices]
    finite = np.isfinite(samples)
    counts = finite.sum(axis=1)
    sums = np.where(finite, samples, 0.0).sum(axis=1)
    means = np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=float),
        where=counts > 0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low = np.nanpercentile(means, 2.5, axis=0)
        high = np.nanpercentile(means, 97.5, axis=0)
    return low, high


def compare_conditions(
    predictions: Iterable[Mapping[str, Any]],
    condition: str,
    baseline: str,
    *,
    reference: str | None = None,
    n_boot: int = 1000,
    seed: int = 4108,
) -> dict[str, Any]:
    """Compare two conditions on exactly equal base/transform support.

    ``delta`` is always ``condition - baseline``. The base sets must be equal,
    and every base must have the same transform set in both conditions. When
    ``reference`` names a condition, it must have exactly the same base and
    transform support as the two compared conditions. Any mismatch is rejected
    before aggregation; support is never silently intersected. The bootstrap
    resamples whole bases, so every transform of a selected base stays
    together and bases remain equally weighted.

    When ``reference`` names a condition, teacher prediction fidelity is added:
    the fraction of rows whose canonical prediction equals the reference
    condition's prediction for the same base and transform.

    Returns ``metrics`` keyed by metric name. Each value contains the paired
    point delta, percentile CI bounds, and the number of bases contributing a
    finite delta. Metric keys are ``BASE_METRIC_KEYS`` plus
    ``FIDELITY_METRIC_KEYS`` when a reference is supplied.
    """
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    grouped = group_by_condition_and_base(predictions)
    for name in (condition, baseline):
        if name not in grouped:
            raise ValueError(f"condition {name!r} is absent from the predictions")
    if reference is not None and reference not in grouped:
        raise ValueError(
            f"reference condition {reference!r} is absent from the predictions"
        )
    if condition == baseline:
        raise ValueError("condition and baseline must differ")

    condition_bases = grouped[condition]
    baseline_bases = grouped[baseline]
    compared_bases = sorted(condition_bases)
    condition_base_ids = set(compared_bases)
    baseline_base_ids = set(baseline_bases)
    if condition_base_ids != baseline_base_ids:
        raise ValueError(
            "condition and baseline must have identical base support; "
            f"missing from {condition!r}: "
            f"{sorted(baseline_base_ids - condition_base_ids)}; "
            f"missing from {baseline!r}: "
            f"{sorted(condition_base_ids - baseline_base_ids)}"
        )
    if reference is not None:
        reference_bases = grouped[reference]
        reference_base_ids = set(reference_bases)
        if reference_base_ids != condition_base_ids:
            raise ValueError(
                "reference support must match the compared conditions; "
                f"missing from {reference!r}: "
                f"{sorted(condition_base_ids - reference_base_ids)}; "
                "missing from the compared conditions: "
                f"{sorted(reference_base_ids - condition_base_ids)}"
            )
    else:
        reference_bases = None

    metric_keys = list(BASE_METRIC_KEYS)
    if reference is not None:
        metric_keys.extend(FIDELITY_METRIC_KEYS)
    metric_index = {key: index for index, key in enumerate(metric_keys)}
    values = np.full((len(compared_bases), len(metric_keys)), np.nan)
    support: dict[str, list[str]] = {}

    for row_index, base_id in enumerate(compared_bases):
        condition_rows = condition_bases[base_id]["rows"]
        baseline_rows = baseline_bases[base_id]["rows"]
        transforms = sorted(condition_rows)
        baseline_transforms = set(baseline_rows)
        if set(transforms) != baseline_transforms:
            raise ValueError(
                f"base {base_id!r} must have identical transform support; "
                f"missing from {condition!r}: "
                f"{sorted(baseline_transforms - set(transforms))}; "
                f"missing from {baseline!r}: "
                f"{sorted(set(transforms) - baseline_transforms)}"
            )
        if reference is not None:
            reference_rows = reference_bases[base_id]["rows"]
            reference_transforms = set(reference_rows)
            if set(transforms) != reference_transforms:
                raise ValueError(
                    f"base {base_id!r} has incomplete reference support; "
                    f"missing from {reference!r}: "
                    f"{sorted(set(transforms) - reference_transforms)}; "
                    "missing from the compared conditions: "
                    f"{sorted(reference_transforms - set(transforms))}"
                )
        else:
            reference_rows = None
        if ORIGINAL_TRANSFORM not in transforms:
            raise ValueError(
                f"base {base_id!r} lacks an original transform"
            )
        support[base_id] = transforms

        condition_stats = _base_metrics(condition_bases[base_id])
        baseline_stats = _base_metrics(baseline_bases[base_id])
        condition_values = _base_comparison_values(condition_stats)
        baseline_values = _base_comparison_values(baseline_stats)

        if reference is not None:
            condition_fidelity = _fidelity_values(
                condition_rows, reference_rows, transforms
            )
            baseline_fidelity = _fidelity_values(
                baseline_rows, reference_rows, transforms
            )
        else:
            condition_fidelity = {}
            baseline_fidelity = {}

        for key in BASE_METRIC_KEYS:
            left = condition_values[key]
            right = baseline_values[key]
            if left is not None and right is not None:
                values[row_index, metric_index[key]] = left - right
        for key in FIDELITY_METRIC_KEYS:
            if reference is None:
                continue
            left = condition_fidelity[key]
            right = baseline_fidelity[key]
            if left is not None and right is not None:
                values[row_index, metric_index[key]] = left - right

    point, counts = _point_means(values)
    ci_low, ci_high = _bootstrap_intervals(values, n_boot, seed)

    metrics: dict[str, dict[str, Any]] = {}
    for key in metric_keys:
        index = metric_index[key]
        metrics[key] = {
            "delta": float(point[index]) if counts[index] else None,
            "ci_low": (
                float(ci_low[index])
                if counts[index] and np.isfinite(ci_low[index])
                else None
            ),
            "ci_high": (
                float(ci_high[index])
                if counts[index] and np.isfinite(ci_high[index])
                else None
            ),
            "n_bases": int(counts[index]),
        }

    return {
        "condition": condition,
        "baseline": baseline,
        "reference": reference,
        "direction": "condition_minus_baseline",
        "n_bases_compared": len(compared_bases),
        "n_bases_only_condition": len(set(condition_bases) - set(baseline_bases)),
        "n_bases_only_baseline": len(set(baseline_bases) - set(condition_bases)),
        "support": support,
        "bootstrap": {
            "unit": "base_id",
            "samples": n_boot,
            "seed": seed,
            "estimator": "base_macro_paired_delta",
        },
        "metrics": metrics,
    }


def _required_number(candidate: Mapping[str, Any], key: str) -> float:
    if key not in candidate or candidate[key] is None:
        raise ValueError(f"lambda candidate is missing required metric {key!r}")
    return float(candidate[key])


def select_lambda(
    candidates: Iterable[Mapping[str, Any]],
    augmented_overall_accuracy: float,
) -> dict[str, Any]:
    """Apply the frozen development lambda-selection rule.

    Each candidate must provide these keys:

    ``lambda``
    ``overall_accuracy``
    ``held_out_jointly_correct_rate``
    ``held_out_consistency``

    Candidates whose overall accuracy is at least
    ``augmented_overall_accuracy - 0.01`` are eligible. Among eligible
    candidates, choose the highest held-out joint correctness, then the highest
    held-out consistency, then the smaller lambda. If none is eligible, choose
    the highest overall accuracy, then joint correctness, consistency, and
    finally the smaller lambda.

    The return value contains the selected candidate, the mode, the threshold,
    and copies of every candidate outcome, so callers can retain all candidate
    results. The function never receives or inspects final-split outcomes.
    """
    parsed: list[dict[str, Any]] = []
    seen_lambdas: set[float] = set()
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        lambda_value = _required_number(candidate, "lambda")
        if lambda_value in seen_lambdas:
            raise ValueError(f"duplicate lambda candidate: {lambda_value!r}")
        seen_lambdas.add(lambda_value)
        for key in SELECTION_KEYS[1:]:
            _required_number(candidate, key)
        parsed.append(candidate)

    if not parsed:
        raise ValueError("no lambda candidates supplied")

    threshold = float(augmented_overall_accuracy) - 0.01
    eligible = [
        candidate
        for candidate in parsed
        if float(candidate["overall_accuracy"]) >= threshold
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda candidate: (
                -float(candidate["held_out_jointly_correct_rate"]),
                -float(candidate["held_out_consistency"]),
                float(candidate["lambda"]),
            ),
        )
        mode = "accuracy_constraint"
    else:
        selected = min(
            parsed,
            key=lambda candidate: (
                -float(candidate["overall_accuracy"]),
                -float(candidate["held_out_jointly_correct_rate"]),
                -float(candidate["held_out_consistency"]),
                float(candidate["lambda"]),
            ),
        )
        mode = "fallback"

    return {
        "selected": dict(selected),
        "selected_lambda": float(selected["lambda"]),
        "mode": mode,
        "accuracy_threshold": threshold,
        "candidates": [dict(candidate) for candidate in parsed],
    }
