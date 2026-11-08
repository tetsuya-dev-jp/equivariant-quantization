#!/usr/bin/env python3
"""Plot final held-out results and write auditable summary tables.

The script consumes the curated report outputs for one model per input directory:
``summaries.json``, ``comparisons.json``, and ``predictions.jsonl.gz``. Figure 1
uses the saved paired base-bootstrap intervals; it never resamples seeds or
treats fit seeds as evaluation bases. Figure 2 and the tables use the public
base-macro study metrics on the actual prediction rows.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eq_quant.study_metrics import (
    HELD_OUT_TRANSFORMS,
    group_by_condition_and_base,
    summarize_conditions,
)

REPORT_FORMAT = "eq_quant_study_report_v1"
FINAL_SPLIT = "final"
STRUCTURAL_SPLIT = "structural_holdout"
FIT_SEEDS = (17, 29, 43)
MODEL_KEYS = ("qwen", "smol")
MODEL_LABELS = {
    "qwen": "Qwen2.5-1.5B",
    "smol": "SmolLM2-1.7B",
}
MODEL_COLORS = {
    "qwen": "#0072B2",
    "smol": "#D55E00",
}
BASELINE_CONDITIONS = (
    ("fp", "fp"),
    ("w8", "8"),
    ("w4", "4"),
    ("w3", "3"),
)
LEARNED_OBJECTIVES = (
    "original_kd",
    "augmented_kd",
    "paired",
    "random_pair",
    "kernel",
)
CONSTANT_CONDITION = "constant_display_A"
RTN_BIAS_CONDITION = "rtn_w4_logit_bias"
RTN_BIAS_OBJECTIVE = "mean_logit_drift"
FIGURE1_METRICS = (
    ("held_out_consistency", "Held-out consistency"),
    ("held_out_jointly_correct_rate", "Held-out jointly correct"),
    (
        "held_out_consistent_wrong_rate",
        "Held-out consistently wrong (lower is better)",
    ),
)
COMPOSITION_METRICS = (
    ("composition_consistency", "Composition consistency"),
    ("composition_jointly_correct_rate", "Composition jointly correct"),
    (
        "composition_consistent_wrong_rate",
        "Composition consistently wrong (lower is better)",
    ),
)
FIGURE2_CONDITIONS = (
    ("fp", "BF16 (fp)"),
    ("w4", "RTN (w4)"),
    (RTN_BIAS_CONDITION, "RTN+bias (w4)"),
    ("original_kd", "original KD"),
    ("augmented_kd", "augmented KD"),
    ("paired", "paired"),
    ("random_pair", "random pair"),
    ("kernel", "kernel"),
    (CONSTANT_CONDITION, "constant A"),
)
FIGURE2_CATEGORIES = (
    ("jointly_correct", "Jointly correct", "#3B6EA5", ""),
    ("consistently_wrong", "Consistently wrong", "#D08A4C", "///"),
    ("inconsistent", "Inconsistent", "#D9D9D9", ".."),
)
TABLE_METRICS = (
    ("held_out_original_accuracy", "original_accuracy"),
    ("held_out_variant_accuracy", "variant_accuracy"),
    ("held_out_consistency", "consistency"),
    ("held_out_jointly_correct", "jointly_correct"),
    ("held_out_consistent_wrong", "consistent_wrong"),
    ("composition_accuracy", "composition_accuracy"),
    ("composition_consistency", "composition_consistency"),
    ("held_out_teacher_fidelity", "teacher_fidelity"),
)
TABLE_STATISTICS = ("mean", "min", "max")
TABLE_HEADER = [
    "model",
    "split",
    "kind",
    "condition",
    "objective",
    "bit_width",
    "fit_count",
    "fit_seeds",
    "bases",
]
for prefix, _ in TABLE_METRICS:
    TABLE_HEADER.extend(f"{prefix}_{statistic}" for statistic in TABLE_STATISTICS)
TABLE_HEADER.extend(
    [
        "held_out_teacher_fidelity_bases",
        "held_out_teacher_fidelity_pairs",
    ]
)
FIGURE1_FILENAME = "final_paired_vs_augmented_heldout"
COMPOSITION_FILENAME = "final_paired_vs_augmented_compositions"
FIGURE2_FILENAME = "final_heldout_outcomes"
FIGURE_SIZE = (7.0, 3.5)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"input file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"prediction file is missing: {path}")
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    return rows


def require_finite(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is missing")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def require_bases(value: Any, label: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} is missing")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be positive, got {number}")
    return number


def validate_report_payload(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("format") != REPORT_FORMAT:
        raise ValueError(f"{label} has an unsupported format: {payload.get('format')!r}")
    if payload.get("state") != "completed":
        raise ValueError(f"{label} is not completed: {payload.get('state')!r}")


def load_model_bundle(model_key: str, directory: Path) -> dict[str, Any]:
    paths = {
        "summaries": directory / "summaries.json",
        "comparisons": directory / "comparisons.json",
        "predictions": directory / "predictions.jsonl.gz",
    }
    summaries = load_json(paths["summaries"])
    comparisons = load_json(paths["comparisons"])
    validate_report_payload(summaries, f"{model_key} summaries")
    validate_report_payload(comparisons, f"{model_key} comparisons")

    summary_block = summaries.get("summaries")
    if not isinstance(summary_block, dict) or not summary_block:
        raise ValueError(f"{model_key} summaries lack condition summaries")
    for condition, summary in summary_block.items():
        require_bases(summary.get("n_bases"), f"{model_key}/{condition} n_bases")

    by_split = summaries.get("grouped_summaries", {}).get("by_split", {})
    for split in (FINAL_SPLIT, STRUCTURAL_SPLIT):
        block = by_split.get(split)
        if not isinstance(block, dict) or not block:
            raise ValueError(f"{model_key} summaries lack grouped split {split!r}")

    comparison_block = comparisons.get("comparisons")
    if not isinstance(comparison_block, list) or not comparison_block:
        raise ValueError(f"{model_key} comparisons lack comparison rows")
    secondary_compositions = comparisons.get("secondary_compositions")
    if not isinstance(secondary_compositions, list) or not secondary_compositions:
        raise ValueError(
            f"{model_key} comparisons lack secondary_compositions rows"
        )

    return {
        "model_key": model_key,
        "directory": directory,
        "paths": paths,
        "summaries": summaries,
        "comparisons": comparisons,
        "rows": load_predictions(paths["predictions"]),
    }


def required_conditions() -> list[str]:
    conditions = [condition for condition, _ in BASELINE_CONDITIONS]
    for objective in LEARNED_OBJECTIVES:
        conditions.extend(f"{objective}_s{seed}" for seed in FIT_SEEDS)
    conditions.append(CONSTANT_CONDITION)
    conditions.append(RTN_BIAS_CONDITION)
    return conditions


def held_out_fidelity(
    grouped: Mapping[str, Mapping[str, Mapping[str, Any]]],
    condition: str,
    reference: str = "fp",
) -> tuple[float, int, int]:
    """Return absolute base-macro held-out agreement with the reference.

    Agreement is computed within each base over exactly the held-out singleton
    transforms shared with the reference, then averaged over bases. The
    reference and condition must have identical base and transform support.
    """
    if condition not in grouped:
        raise ValueError(f"condition {condition!r} is absent from predictions")
    if reference not in grouped:
        raise ValueError(f"reference {reference!r} is absent from predictions")

    condition_bases = grouped[condition]
    reference_bases = grouped[reference]
    condition_ids = set(condition_bases)
    reference_ids = set(reference_bases)
    if condition_ids != reference_ids:
        raise ValueError(
            f"condition {condition!r} and reference {reference!r} have "
            "different base support"
        )

    per_base: list[float] = []
    n_pairs = 0
    for base_id in sorted(condition_ids):
        condition_rows = condition_bases[base_id]["rows"]
        reference_rows = reference_bases[base_id]["rows"]
        condition_transforms = sorted(
            transform
            for transform in condition_rows
            if transform in HELD_OUT_TRANSFORMS
        )
        reference_transforms = sorted(
            transform
            for transform in reference_rows
            if transform in HELD_OUT_TRANSFORMS
        )
        if condition_transforms != reference_transforms:
            raise ValueError(
                f"base {base_id!r} has different held-out support between "
                f"{condition!r} and {reference!r}"
            )
        if not condition_transforms:
            continue
        matches = sum(
            int(
                condition_rows[transform]["canonical_prediction"]
                == reference_rows[transform]["canonical_prediction"]
            )
            for transform in condition_transforms
        )
        per_base.append(matches / len(condition_transforms))
        n_pairs += len(condition_transforms)

    if not per_base:
        raise ValueError(f"condition {condition!r} has no held-out singleton rows")
    return sum(per_base) / len(per_base), len(per_base), n_pairs


def condition_record(
    condition: str,
    summary: Mapping[str, Any],
    fidelity: tuple[float, int, int],
) -> dict[str, Any]:
    held_out = summary.get("endpoints", {}).get("held_out_singles")
    compositions = summary.get("endpoints", {}).get("compositions")
    if not isinstance(held_out, dict):
        raise ValueError(f"{condition} lacks held-out endpoint metrics")
    if not isinstance(compositions, dict):
        raise ValueError(f"{condition} lacks composition endpoint metrics")

    teacher_fidelity, fidelity_bases, fidelity_pairs = fidelity
    return {
        "condition": condition,
        "n_bases": require_bases(summary.get("n_bases"), f"{condition} n_bases"),
        "original_accuracy": require_finite(
            summary.get("original_accuracy_macro"),
            f"{condition} original accuracy",
        ),
        "variant_accuracy": require_finite(
            held_out.get("accuracy_macro"),
            f"{condition} held-out variant accuracy",
        ),
        "consistency": require_finite(
            held_out.get("consistency_macro"),
            f"{condition} held-out consistency",
        ),
        "jointly_correct": require_finite(
            held_out.get("jointly_correct_rate_macro"),
            f"{condition} held-out jointly-correct rate",
        ),
        "consistent_wrong": require_finite(
            held_out.get("consistent_wrong_rate_macro"),
            f"{condition} held-out consistently-wrong rate",
        ),
        "composition_accuracy": require_finite(
            compositions.get("accuracy_macro"),
            f"{condition} composition accuracy",
        ),
        "composition_consistency": require_finite(
            compositions.get("consistency_macro"),
            f"{condition} composition consistency",
        ),
        "teacher_fidelity": require_finite(
            teacher_fidelity,
            f"{condition} held-out teacher fidelity",
        ),
        "fidelity_bases": require_bases(
            fidelity_bases, f"{condition} teacher fidelity bases"
        ),
        "fidelity_pairs": require_bases(
            fidelity_pairs, f"{condition} teacher fidelity pairs"
        ),
    }


def validate_seed_support(
    grouped: Mapping[str, Mapping[str, Mapping[str, Any]]],
    model_key: str,
    split: str,
) -> None:
    for objective in LEARNED_OBJECTIVES:
        base_sets: list[set[str]] = []
        for seed in FIT_SEEDS:
            condition = f"{objective}_s{seed}"
            if condition not in grouped:
                raise ValueError(f"{model_key}/{split} lacks {condition}")
            base_sets.append(set(grouped[condition]))
        if any(base_set != base_sets[0] for base_set in base_sets[1:]):
            raise ValueError(
                f"{model_key}/{split} objective {objective!r} has different "
                "base support across fit seeds"
            )
        if not base_sets[0]:
            raise ValueError(
                f"{model_key}/{split} objective {objective!r} has no bases"
            )


def compute_split_metrics(
    model_key: str,
    rows: Sequence[Mapping[str, Any]],
    split: str,
) -> dict[str, Any]:
    selected = [dict(row) for row in rows if str(row["split"]) == split]
    if not selected:
        raise ValueError(f"{model_key} has no prediction rows for {split!r}")

    conditions = required_conditions()
    available = {str(row["condition"]) for row in selected}
    missing = [condition for condition in conditions if condition not in available]
    if missing:
        raise ValueError(
            f"{model_key}/{split} is missing conditions: {sorted(missing)}"
        )

    summaries = summarize_conditions(selected, conditions)
    grouped = group_by_condition_and_base(selected)
    validate_seed_support(grouped, model_key, split)

    records = {}
    for condition in conditions:
        summary = summaries[condition]
        require_bases(summary.get("n_bases"), f"{model_key}/{split}/{condition} n_bases")
        records[condition] = condition_record(
            condition,
            summary,
            held_out_fidelity(grouped, condition),
        )

    return {
        "split": split,
        "summaries": summaries,
        "grouped": grouped,
        "records": records,
    }


def comparison_metric(
    comparisons: Sequence[Mapping[str, Any]],
    seed: int,
    metric: str,
    *,
    evidence: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    condition = f"paired_s{seed}"
    baseline = f"augmented_kd_s{seed}"
    matches = [
        entry
        for entry in comparisons
        if entry.get("split") == FINAL_SPLIT
        and entry.get("task_family") == "all"
        and entry.get("evidence") == evidence
        and entry.get("fit_seed") == seed
        and entry.get("condition") == condition
        and entry.get("baseline") == baseline
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one final/all/all {evidence} paired-vs-augmented "
            f"comparison for seed {seed}, found {len(matches)}"
        )
    entry = matches[0]
    comparison = entry.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError(f"comparison for seed {seed} lacks a comparison payload")
    if comparison.get("condition") != condition:
        raise ValueError(f"comparison for seed {seed} has a condition mismatch")
    if comparison.get("baseline") != baseline:
        raise ValueError(f"comparison for seed {seed} has a baseline mismatch")
    if comparison.get("direction") != "condition_minus_baseline":
        raise ValueError(f"comparison for seed {seed} has an unexpected direction")
    metrics = comparison.get("metrics")
    if not isinstance(metrics, dict) or metric not in metrics:
        raise ValueError(f"comparison for seed {seed} lacks metric {metric!r}")
    return entry, metrics[metric]


def collect_effect_rows(
    model_key: str,
    comparisons: Sequence[Mapping[str, Any]],
    final_metrics: Mapping[str, Any],
    *,
    evidence: str,
    metrics_spec: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in FIT_SEEDS:
        condition = f"paired_s{seed}"
        baseline = f"augmented_kd_s{seed}"
        metrics: dict[str, Any] = {}
        for metric, _ in metrics_spec:
            entry, metric_payload = comparison_metric(
                comparisons, seed, metric, evidence=evidence
            )
            comparison = entry["comparison"]
            delta = require_finite(
                metric_payload.get("delta"),
                f"{model_key}/seed{seed}/{metric} delta",
            )
            ci_low = require_finite(
                metric_payload.get("ci_low"),
                f"{model_key}/seed{seed}/{metric} ci_low",
            )
            ci_high = require_finite(
                metric_payload.get("ci_high"),
                f"{model_key}/seed{seed}/{metric} ci_high",
            )
            n_bases = require_bases(
                metric_payload.get("n_bases"),
                f"{model_key}/seed{seed}/{metric} n_bases",
            )
            if not ci_low <= delta <= ci_high:
                raise ValueError(
                    f"{model_key}/seed{seed}/{metric} has a CI that excludes "
                    "its point estimate"
                )
            bootstrap = comparison.get("bootstrap")
            if not isinstance(bootstrap, dict):
                raise ValueError(
                    f"{model_key}/seed{seed}/{metric} lacks bootstrap provenance"
                )
            if bootstrap.get("unit") != "base_id":
                raise ValueError(
                    f"{model_key}/seed{seed}/{metric} is not a base bootstrap"
                )
            if int(bootstrap.get("samples", 0)) <= 0:
                raise ValueError(
                    f"{model_key}/seed{seed}/{metric} has no bootstrap samples"
                )
            summary_bases = final_metrics["records"][condition]["n_bases"]
            baseline_bases = final_metrics["records"][baseline]["n_bases"]
            if n_bases != summary_bases or n_bases != baseline_bases:
                raise ValueError(
                    f"{model_key}/seed{seed}/{metric} n_bases does not match "
                    "the final summaries"
                )
            metrics[metric] = {
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "delta_pp": 100.0 * delta,
                "ci_low_pp": 100.0 * ci_low,
                "ci_high_pp": 100.0 * ci_high,
                "n_bases": n_bases,
                "bootstrap": {
                    "unit": bootstrap.get("unit"),
                    "samples": int(bootstrap.get("samples")),
                    "seed": bootstrap.get("seed"),
                },
            }
        rows.append(
            {
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "seed": seed,
                "condition": condition,
                "baseline": baseline,
                "metrics": metrics,
            }
        )
    return rows


def summarize_values(values: Sequence[float], label: str) -> dict[str, float]:
    if not values:
        raise ValueError(f"{label} has no values")
    finite = [require_finite(value, label) for value in values]
    return {
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "max": max(finite),
    }


def make_table_row(
    model_key: str,
    split: str,
    kind: str,
    condition: str,
    objective: str,
    bit_width: str,
    fit_count: int,
    fit_seeds: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"{model_key}/{split}/{condition} has no outcomes")
    base_counts = {int(record["n_bases"]) for record in records}
    if len(base_counts) != 1:
        raise ValueError(
            f"{model_key}/{split}/{condition} has inconsistent base counts"
        )
    fidelity_bases = {int(record["fidelity_bases"]) for record in records}
    fidelity_pairs = {int(record["fidelity_pairs"]) for record in records}
    if len(fidelity_bases) != 1 or len(fidelity_pairs) != 1:
        raise ValueError(
            f"{model_key}/{split}/{condition} has inconsistent fidelity support"
        )

    row: dict[str, Any] = {
        "model": model_key,
        "split": split,
        "kind": kind,
        "condition": condition,
        "objective": objective,
        "bit_width": bit_width,
        "fit_count": fit_count,
        "fit_seeds": fit_seeds,
        "bases": base_counts.pop(),
        "held_out_teacher_fidelity_bases": fidelity_bases.pop(),
        "held_out_teacher_fidelity_pairs": fidelity_pairs.pop(),
    }
    for prefix, key in TABLE_METRICS:
        statistics = summarize_values(
            [record[key] for record in records],
            f"{model_key}/{split}/{condition}/{key}",
        )
        for statistic in TABLE_STATISTICS:
            row[f"{prefix}_{statistic}"] = statistics[statistic]
    return row


def build_table_rows(
    model_key: str,
    split: str,
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = metrics["records"]
    rows: list[dict[str, Any]] = []
    for condition, bit_width in BASELINE_CONDITIONS:
        rows.append(
            make_table_row(
                model_key,
                split,
                "baseline",
                condition,
                "baseline",
                bit_width,
                0,
                "",
                [records[condition]],
            )
        )
    for objective in LEARNED_OBJECTIVES:
        seed_records = [
            records[f"{objective}_s{seed}"] for seed in FIT_SEEDS
        ]
        rows.append(
            make_table_row(
                model_key,
                split,
                "fitted",
                objective,
                objective,
                "",
                len(seed_records),
                "17;29;43",
                seed_records,
            )
        )
    rows.append(
        make_table_row(
            model_key,
            split,
            "untrained_rule",
            CONSTANT_CONDITION,
            CONSTANT_CONDITION,
            "",
            0,
            "None",
            [records[CONSTANT_CONDITION]],
        )
    )
    rows.append(
        make_table_row(
            model_key,
            split,
            "control_rule",
            RTN_BIAS_CONDITION,
            RTN_BIAS_OBJECTIVE,
            "4",
            0,
            "None",
            [records[RTN_BIAS_CONDITION]],
        )
    )
    return rows


def composition_categories(record: Mapping[str, Any], label: str) -> dict[str, float]:
    consistency = require_finite(record["consistency"], f"{label} consistency")
    jointly_correct = require_finite(
        record["jointly_correct"], f"{label} jointly correct"
    )
    consistently_wrong = require_finite(
        record["consistent_wrong"], f"{label} consistently wrong"
    )
    inconsistent = 1.0 - consistency
    categories = {
        "jointly_correct": jointly_correct,
        "consistently_wrong": consistently_wrong,
        "inconsistent": inconsistent,
    }
    for name, value in categories.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} {name} is outside [0, 1]: {value}")
    total = sum(categories.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} outcome categories sum to {total} instead of 1")
    return categories


def build_figure2_panel(
    model_key: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    records = metrics["records"]
    panel: dict[str, Any] = {}
    for condition, label in FIGURE2_CONDITIONS:
        if condition in {name for name, _ in BASELINE_CONDITIONS}:
            condition_names = [condition]
            seeds: list[int | None] = [None]
            fit_count = 0
        elif condition in {CONSTANT_CONDITION, RTN_BIAS_CONDITION}:
            condition_names = [condition]
            seeds = [None]
            fit_count = 0
        else:
            condition_names = [
                f"{condition}_s{seed}" for seed in FIT_SEEDS
            ]
            seeds = list(FIT_SEEDS)
            fit_count = len(seeds)

        outcomes = []
        for seed, condition_name in zip(seeds, condition_names):
            record = records[condition_name]
            categories = composition_categories(
                record, f"{model_key}/{metrics['split']}/{condition_name}"
            )
            outcomes.append(
                {
                    "seed": seed,
                    "condition": condition_name,
                    "n_bases": int(record["n_bases"]),
                    **categories,
                }
            )

        if condition == RTN_BIAS_CONDITION:
            if fit_count != 0 or seeds != [None] or len(outcomes) != 1:
                raise ValueError(
                    f"{model_key}/{metrics['split']}/{condition} must be one "
                    "deterministic control outcome, not a seed fit"
                )
        base_counts = {outcome["n_bases"] for outcome in outcomes}
        if len(base_counts) != 1:
            raise ValueError(
                f"{model_key}/{metrics['split']}/{condition} has different "
                "base counts across outcomes"
            )
        mean = {
            key: sum(outcome[key] for outcome in outcomes) / len(outcomes)
            for key, _, _, _ in FIGURE2_CATEGORIES
        }
        total = sum(mean.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"{model_key}/{metrics['split']}/{condition} mean outcome "
                f"categories sum to {total} instead of 1"
            )
        panel[condition] = {
            "label": label,
            "fit_count": fit_count,
            "n_bases": base_counts.pop(),
            "outcomes": outcomes,
            "mean": mean,
        }
    return panel


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(color="#D9D9D9", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_effects(
    effect_rows: Sequence[Mapping[str, Any]],
    metrics_spec: Sequence[tuple[str, str]],
    output_dir: Path,
    stem: str,
) -> None:
    fig, axes = plt.subplots(
        1, len(metrics_spec), figsize=FIGURE_SIZE, sharey=True
    )
    positions = {
        (row["model"], row["seed"]): len(effect_rows) - 1 - index
        for index, row in enumerate(effect_rows)
    }
    all_limits = [0.0]
    for ax, (metric, label) in zip(axes, metrics_spec):
        for row in effect_rows:
            payload = row["metrics"][metric]
            value = payload["delta_pp"]
            low = payload["ci_low_pp"]
            high = payload["ci_high_pp"]
            all_limits.extend([low, high])
            ax.errorbar(
                [value],
                [positions[(row["model"], row["seed"])]],
                xerr=[[value - low], [high - value]],
                fmt="o",
                color=MODEL_COLORS[row["model"]],
                markerfacecolor=MODEL_COLORS[row["model"]],
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=5.0,
                elinewidth=1.0,
                capsize=2.5,
                zorder=3,
            )
        ax.axvline(0.0, color="#333333", linewidth=1.0, zorder=1)
        ax.axhline(2.5, color="#D9D9D9", linewidth=0.7, zorder=1)
        ax.set_xlabel("Paired − augmented KD (percentage points)")
        ax.set_title(label, fontsize=8, fontweight="bold", color="#222222")
        style_axes(ax)

    lower = min(all_limits) - 2.0
    upper = max(all_limits) + 2.0
    for ax in axes:
        ax.set_xlim(lower, upper)
        ax.set_ylim(-0.6, len(effect_rows) - 0.4)

    tick_positions = [
        positions[(row["model"], row["seed"])] for row in effect_rows
    ]
    tick_labels = [
        f"{MODEL_LABELS[row['model']]} s{row['seed']}" for row in effect_rows
    ]
    axes[0].set_yticks(tick_positions)
    axes[0].set_yticklabels(tick_labels)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[key],
            marker="o",
            linestyle="",
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=MODEL_LABELS[key],
        )
        for key in MODEL_KEYS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=8,
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.20, top=0.80, wspace=0.12)
    save_figure(fig, output_dir, stem)


def plot_figure2(
    panels: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)
    condition_keys = [condition for condition, _ in FIGURE2_CONDITIONS]
    x = list(range(len(condition_keys)))
    for ax, model_key in zip(axes, MODEL_KEYS):
        panel = panels[model_key]
        bottoms = [0.0] * len(condition_keys)
        for key, label, color, hatch in FIGURE2_CATEGORIES:
            values = [panel[condition]["mean"][key] for condition in condition_keys]
            ax.bar(
                x,
                values,
                bottom=bottoms,
                width=0.62,
                color=color,
                edgecolor="#333333",
                linewidth=0.6,
                hatch=hatch,
                label=label,
                zorder=3,
            )
            bottoms = [
                bottom + value for bottom, value in zip(bottoms, values)
            ]
        ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [label for _, label in FIGURE2_CONDITIONS],
            rotation=32,
            ha="right",
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        style_axes(ax)
        ax.text(
            0.0,
            1.04,
            f"{'A' if model_key == 'qwen' else 'B'}  {MODEL_LABELS[model_key]}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#222222",
        )
    axes[0].set_ylabel("Proportion of held-out pairs")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=8,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.27, top=0.80, wspace=0.10)
    save_figure(fig, output_dir, FIGURE2_FILENAME)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf")
    fig.savefig(output_dir / f"{stem}.png", format="png", dpi=300)
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in TABLE_HEADER})


def write_plot_data(
    path: Path,
    bundles: Mapping[str, Mapping[str, Any]],
    figure1_rows: Sequence[Mapping[str, Any]],
    composition_rows: Sequence[Mapping[str, Any]],
    figure2_panels: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = {
        "provenance": {
            "script": "scripts/plot_final.py",
            "inputs": {
                key: {
                    "directory": str(bundle["directory"]),
                    "summaries": str(bundle["paths"]["summaries"]),
                    "comparisons": str(bundle["paths"]["comparisons"]),
                    "predictions": str(bundle["paths"]["predictions"]),
                }
                for key, bundle in bundles.items()
            },
            "fit_seeds": list(FIT_SEEDS),
            "figure1": {
                "split": FINAL_SPLIT,
                "task_family": "all",
                "evidence": "held_out_singles",
                "condition": "paired_s<seed>",
                "baseline": "augmented_kd_s<seed>",
                "units": "percentage points of paired minus augmented KD",
                "intervals": "saved 95% base-bootstrap CIs; no seed averaging",
                "selection_note": (
                    "held-out singles are gradient-held-out but were used for "
                    "development selection; they are not compositions"
                ),
            },
            "figure_compositions": {
                "split": FINAL_SPLIT,
                "task_family": "all",
                "evidence": "compositions",
                "condition": "paired_s<seed>",
                "baseline": "augmented_kd_s<seed>",
                "units": "percentage points of paired minus augmented KD",
                "intervals": "saved 95% base-bootstrap CIs; no seed averaging",
                "selection_note": (
                    "compositions are withheld from both fitting and selection"
                ),
            },
            "figure2": {
                "split": FINAL_SPLIT,
                "endpoint": "held_out_singles",
                "unit": "base-macro proportion",
                "aggregation": (
                    "learned objectives: mean over fit seeds 17, 29, 43; "
                    "baseline and control rules: one outcome"
                ),
                "inconsistent": "1 - held_out consistency",
            },
            "table": {
                "splits": [FINAL_SPLIT, STRUCTURAL_SPLIT],
                "statistics": list(TABLE_STATISTICS),
                "fidelity": (
                    "absolute held-out agreement with fp, averaged within "
                    "each base and then over bases; fit seeds averaged last"
                ),
                "confidence_intervals": "none; only across-seed min/max",
                "control_conditions": {
                    CONSTANT_CONDITION: {
                        "objective": CONSTANT_CONDITION,
                        "kind": "untrained_rule",
                        "fit_seed": None,
                        "fit_count": 0,
                    },
                    RTN_BIAS_CONDITION: {
                        "objective": RTN_BIAS_OBJECTIVE,
                        "kind": "control_rule",
                        "fit_seed": None,
                        "fit_count": 0,
                        "parameters": 4,
                    },
                },
            },
            "bootstrap": {
                key: bundle["comparisons"].get("bootstrap")
                for key, bundle in bundles.items()
            },
        },
        "figure1": {
            "metric_labels": {
                metric: label for metric, label in FIGURE1_METRICS
            },
            "rows": list(figure1_rows),
        },
        "figure_compositions": {
            "metric_labels": {
                metric: label for metric, label in COMPOSITION_METRICS
            },
            "rows": list(composition_rows),
        },
        "figure2": {
            "categories": {
                key: label for key, label, _, _ in FIGURE2_CATEGORIES
            },
            "panels": figure2_panels,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(qwen_dir: Path, smol_dir: Path, output_dir: Path) -> list[Path]:
    input_dirs = {"qwen": qwen_dir, "smol": smol_dir}
    bundles = {
        key: load_model_bundle(key, directory)
        for key, directory in input_dirs.items()
    }

    figure1_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    figure2_panels: dict[str, Any] = {}
    table_final: list[dict[str, Any]] = []
    table_structural: list[dict[str, Any]] = []

    for model_key in MODEL_KEYS:
        bundle = bundles[model_key]
        final_metrics = compute_split_metrics(
            model_key, bundle["rows"], FINAL_SPLIT
        )
        structural_metrics = compute_split_metrics(
            model_key, bundle["rows"], STRUCTURAL_SPLIT
        )
        figure1_rows.extend(
            collect_effect_rows(
                model_key,
                bundle["comparisons"]["comparisons"],
                final_metrics,
                evidence="held_out_singles",
                metrics_spec=FIGURE1_METRICS,
            )
        )
        composition_rows.extend(
            collect_effect_rows(
                model_key,
                bundle["comparisons"]["secondary_compositions"],
                final_metrics,
                evidence="compositions",
                metrics_spec=COMPOSITION_METRICS,
            )
        )
        figure2_panels[model_key] = build_figure2_panel(
            model_key, final_metrics
        )
        table_final.extend(
            build_table_rows(model_key, FINAL_SPLIT, final_metrics)
        )
        table_structural.extend(
            build_table_rows(model_key, STRUCTURAL_SPLIT, structural_metrics)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_effects(
        figure1_rows, FIGURE1_METRICS, output_dir, FIGURE1_FILENAME
    )
    plot_effects(
        composition_rows,
        COMPOSITION_METRICS,
        output_dir,
        COMPOSITION_FILENAME,
    )
    plot_figure2(figure2_panels, output_dir)
    write_plot_data(
        output_dir / "plot-data.json",
        bundles,
        figure1_rows,
        composition_rows,
        figure2_panels,
    )
    write_csv(output_dir / "table-final.csv", table_final)
    write_csv(output_dir / "table-structural.csv", table_structural)

    return [
        output_dir / f"{FIGURE1_FILENAME}.pdf",
        output_dir / f"{FIGURE1_FILENAME}.png",
        output_dir / f"{COMPOSITION_FILENAME}.pdf",
        output_dir / f"{COMPOSITION_FILENAME}.png",
        output_dir / f"{FIGURE2_FILENAME}.pdf",
        output_dir / f"{FIGURE2_FILENAME}.png",
        output_dir / "plot-data.json",
        output_dir / "table-final.csv",
        output_dir / "table-structural.csv",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-dir",
        type=Path,
        default=Path("results/final/qwen"),
        help="directory with the Qwen final summaries, comparisons, and predictions",
    )
    parser.add_argument(
        "--smol-dir",
        type=Path,
        default=Path("results/final/smol"),
        help="directory with the SmolLM2 final summaries, comparisons, and predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/final"),
        help="directory for figures, plot data, and tables",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "hatch.linewidth": 0.6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    for path in run(args.qwen_dir, args.smol_dir, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
