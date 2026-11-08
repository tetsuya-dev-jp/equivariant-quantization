"""Rerun the frozen final report analysis from released predictions only.

The replay reads ``predictions.jsonl.gz``, the released compact ``summaries.json``
and ``comparisons.json`` (and optional ``selection.json``), then reruns the
existing canonical report helpers. It never loads model weights or fits.

Usage::

    python scripts/replay_final.py --source results/final/qwen --output local/replay/qwen
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eq_quant.study import (
    RTN_BIAS_OBJECTIVE,
    _group_summaries,
    _per_transform_evidence,
    _primary_comparisons,
    _without_bases,
)
from eq_quant.study_metrics import summarize_conditions

TOLERANCE = 1e-12
GZIP_PREDICTIONS = "predictions.jsonl.gz"
PLAIN_PREDICTIONS = "predictions.jsonl"
RELEASED_SUMMARIES = "summaries.json"
RELEASED_COMPARISONS = "comparisons.json"
OPTIONAL_SELECTION = "selection.json"

ORDER_INSENSITIVE_LIST_KEYS = frozenset(
    {
        "descriptive_control_conditions",
        "untrained_rule_conditions",
        "secondary_postprocessing_conditions",
    }
)

EXCLUDED_KEYS = frozenset(
    {
        "bases",
        "source",
        "source_revision",
        "source_dirty_paths",
        "provenance",
        "created_at",
        "generated_at",
        "timestamp",
        "paths",
        "path",
        "manifest",
        "optimization_time",
    }
)


class ReplayError(Exception):
    """Raised when released inputs cannot be read or replayed."""


class ReplayMismatch(ReplayError):
    """Raised when regenerated results disagree with the released report."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ReplayError(f"{path} must contain a JSON object")
    return payload


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ReplayError(
                            f"{path} contains a prediction row that is not an object"
                        )
                    rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read predictions at {path}: {error}") from error
    if not rows:
        raise ReplayError(f"no prediction rows at {path}")
    return rows


def _prediction_path(source: Path) -> Path:
    gz_path = source / GZIP_PREDICTIONS
    plain_path = source / PLAIN_PREDICTIONS
    if gz_path.is_file():
        return gz_path
    if plain_path.is_file():
        return plain_path
    raise ReplayError(
        f"predictions are missing from {source}: expected {GZIP_PREDICTIONS} "
        f"or {PLAIN_PREDICTIONS}"
    )


def load_source(source: str | Path) -> dict[str, Any]:
    """Read the released source directory without loading any model artifact."""
    source_path = Path(source)
    if not source_path.is_dir():
        raise ReplayError(f"source directory is missing: {source_path}")
    summaries_path = source_path / RELEASED_SUMMARIES
    comparisons_path = source_path / RELEASED_COMPARISONS
    for path in (summaries_path, comparisons_path):
        if not path.is_file():
            raise ReplayError(f"released report file is missing: {path}")
    selection_path = source_path / OPTIONAL_SELECTION
    return {
        "source": source_path,
        "predictions": _read_predictions(_prediction_path(source_path)),
        "summaries": _read_json(summaries_path),
        "comparisons": _read_json(comparisons_path),
        "selection": (
            _read_json(selection_path) if selection_path.is_file() else None
        ),
    }


def _condition_index(summaries: Mapping[str, Any]) -> dict[str, Any]:
    index = summaries.get("condition_index")
    if not isinstance(index, dict) or not index:
        raise ReplayError("released summaries lack a non-empty condition_index")
    for condition, entry in index.items():
        if not isinstance(entry, dict):
            raise ReplayError(
                f"condition_index entry for {condition!r} is not an object"
            )
    return index


def _sanitized_condition_index(index: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for condition, entry in index.items():
        sanitized[condition] = {
            key: value
            for key, value in entry.items()
            if key not in EXCLUDED_KEYS and not key.endswith("_path")
        }
    return sanitized


def _analysis_config(
    summaries: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    config = summaries.get("config")
    if isinstance(config, dict):
        analysis = config.get("analysis")
        if isinstance(analysis, dict):
            return analysis
    if selection is not None:
        config = selection.get("config")
        if isinstance(config, dict):
            analysis = config.get("analysis")
            if isinstance(analysis, dict):
                return analysis
    return {}


def _resolve_bootstrap(
    comparisons: Mapping[str, Any],
    summaries: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
) -> tuple[int, int]:
    released = comparisons.get("bootstrap")
    fallback = _analysis_config(summaries, selection)
    samples = None
    seed = None
    if isinstance(released, dict):
        samples = released.get("samples")
        seed = released.get("seed")
    if samples is None:
        samples = fallback.get("bootstrap_samples", 1000)
    if seed is None:
        seed = fallback.get("seed", 4108)
    try:
        samples = int(samples)
        seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ReplayError(f"invalid bootstrap settings: {error}") from error
    if samples <= 0:
        raise ReplayError(f"bootstrap sample count must be positive: {samples}")
    return samples, seed


def _model_id(
    summaries: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
) -> str | None:
    for container in (summaries, selection):
        if not isinstance(container, Mapping):
            continue
        model = container.get("model")
        if isinstance(model, dict) and model.get("id"):
            return str(model["id"])
        config = container.get("config")
        if isinstance(config, dict):
            model = config.get("model")
            if isinstance(model, dict) and model.get("id"):
                return str(model["id"])
    return None


def _fit_seeds(
    index: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
) -> list[int]:
    seeds = {
        int(entry["fit_seed"])
        for entry in index.values()
        if entry.get("fit_seed") is not None
    }
    if not seeds and selection is not None:
        seeds = {int(seed) for seed in selection.get("fit_seeds", [])}
    if not seeds:
        raise ReplayError("no fit seeds are available for primary comparisons")
    return sorted(seeds)


def recompute(
    predictions: Iterable[Mapping[str, Any]],
    *,
    condition_index: Mapping[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Regenerate the report analysis with the existing study helpers."""
    rows = [dict(row) for row in predictions]
    conditions = sorted(condition_index)
    present_conditions = {str(row["condition"]) for row in rows}
    if present_conditions != set(conditions):
        missing = sorted(set(conditions) - present_conditions)
        extra = sorted(present_conditions - set(conditions))
        raise ReplayError(
            "prediction conditions do not match condition_index; "
            f"missing from predictions: {missing}; "
            f"missing from condition_index: {extra}"
        )

    summaries = _without_bases(summarize_conditions(rows, conditions))
    splits = sorted({str(row["split"]) for row in rows})
    families = sorted({str(row["task_family"]) for row in rows})
    seed_by_condition = {
        condition: int(entry["fit_seed"])
        for condition, entry in condition_index.items()
        if entry.get("fit_seed") is not None
    }
    grouped = _group_summaries(
        rows, conditions, splits, families, seed_by_condition
    )
    if model_id is not None:
        grouped["by_model"] = {model_id: summaries}
    seeds = _fit_seeds(condition_index, None)
    comparisons = _primary_comparisons(
        rows,
        seeds,
        splits,
        families,
        bootstrap_samples,
        bootstrap_seed,
    )
    secondary_compositions = _primary_comparisons(
        rows,
        seeds,
        splits,
        families,
        bootstrap_samples,
        bootstrap_seed,
        evidence="compositions",
    )
    per_transform = _per_transform_evidence(rows, conditions)
    descriptive_controls = [
        condition
        for condition, entry in condition_index.items()
        if entry.get("objective")
        in {"baseline", "original_kd", "random_pair", "kernel"}
        or entry.get("objective") == RTN_BIAS_OBJECTIVE
        or entry.get("untrained_rule")
    ]
    untrained_rules = [
        condition
        for condition, entry in condition_index.items()
        if entry.get("untrained_rule")
    ]
    secondary_postprocessing = [
        condition
        for condition, entry in condition_index.items()
        if entry.get("secondary_postprocessing")
    ]
    return {
        "conditions": conditions,
        "summaries": summaries,
        "grouped_summaries": grouped,
        "per_transform": per_transform,
        "comparisons": comparisons,
        "secondary_compositions": secondary_compositions,
        "descriptive_control_conditions": descriptive_controls,
        "untrained_rule_conditions": untrained_rules,
        "secondary_postprocessing_conditions": secondary_postprocessing,
        "splits": splits,
        "task_families": families,
    }


def _comparison_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    required = ("fit_seed", "split", "task_family", "condition", "baseline")
    missing = [key for key in required if key not in entry]
    if missing:
        raise ReplayError(
            f"comparison entry is missing key fields {missing}: {entry!r}"
        )
    return (
        int(entry["fit_seed"]),
        str(entry["split"]),
        str(entry["task_family"]),
        str(entry["condition"]),
        str(entry["baseline"]),
        str(entry.get("evidence", "")),
    )


def _sorted_comparisons(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(entry) for entry in entries),
        key=_comparison_key,
    )


def build_payloads(
    released_summaries: Mapping[str, Any],
    released_comparisons: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build compact summaries and comparisons payloads for output."""
    summaries_payload: dict[str, Any] = {}
    for key in ("format", "state", "model", "config"):
        if key in released_summaries:
            summaries_payload[key] = copy.deepcopy(released_summaries[key])
    summaries_payload.update(
        {
            "conditions": recomputed["conditions"],
            "condition_index": _sanitized_condition_index(
                released_summaries["condition_index"]
            ),
            "summaries": recomputed["summaries"],
            "grouped_summaries": recomputed["grouped_summaries"],
            "per_transform": recomputed["per_transform"],
            "descriptive_control_conditions": recomputed[
                "descriptive_control_conditions"
            ],
            "untrained_rule_conditions": recomputed[
                "untrained_rule_conditions"
            ],
            "secondary_postprocessing_conditions": recomputed[
                "secondary_postprocessing_conditions"
            ],
        }
    )

    comparisons_payload: dict[str, Any] = {}
    for key in ("format", "state", "model", "config"):
        if key in released_comparisons:
            comparisons_payload[key] = copy.deepcopy(released_comparisons[key])
    bootstrap = copy.deepcopy(released_comparisons.get("bootstrap") or {})
    bootstrap.update(
        {"unit": "base_id", "samples": bootstrap_samples, "seed": bootstrap_seed}
    )
    comparisons_payload.update(
        {
            "primary": released_comparisons.get(
                "primary", "paired_vs_augmented_kd_same_seed"
            ),
            "evidence": released_comparisons.get(
                "evidence", "held_out_singles"
            ),
            "reference_condition": released_comparisons.get(
                "reference_condition", "fp"
            ),
            "bootstrap": bootstrap,
            "comparisons": _sorted_comparisons(recomputed["comparisons"]),
            "secondary_compositions": _sorted_comparisons(
                recomputed["secondary_compositions"]
            ),
        }
    )
    if "endpoint_descriptions" in released_comparisons:
        comparisons_payload["endpoint_descriptions"] = copy.deepcopy(
            released_comparisons["endpoint_descriptions"]
        )
    return summaries_payload, comparisons_payload


def _excluded(key: str) -> bool:
    return key in EXCLUDED_KEYS or key.endswith("_path")


def _mismatch(
    mismatches: list[str],
    path: str,
    message: str,
) -> None:
    mismatches.append(f"{path}: {message}")


def _describe(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


def _compare_value(
    expected: Any,
    actual: Any,
    path: str,
    mismatches: list[str],
    tolerance: float,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping):
            _mismatch(mismatches, path, f"expected object, found {_describe(actual)}")
            return
        for key, value in expected.items():
            if _excluded(str(key)):
                continue
            child = f"{path}.{key}"
            if key not in actual:
                _mismatch(mismatches, child, "released field is missing from replay")
                continue
            _compare_value(value, actual[key], child, mismatches, tolerance)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            _mismatch(mismatches, path, f"expected list, found {_describe(actual)}")
            return
        if len(expected) != len(actual):
            _mismatch(
                mismatches,
                path,
                f"expected {len(expected)} entries, found {len(actual)}",
            )
        for index in range(min(len(expected), len(actual))):
            _compare_value(
                expected[index],
                actual[index],
                f"{path}[{index}]",
                mismatches,
                tolerance,
            )
        return

    if isinstance(expected, bool):
        if not isinstance(actual, bool) or expected != actual:
            _mismatch(mismatches, path, f"expected {expected!r}, found {_describe(actual)}")
        return

    if isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int):
            _mismatch(mismatches, path, f"expected integer, found {_describe(actual)}")
        elif expected != actual:
            _mismatch(mismatches, path, f"expected {expected}, found {actual}")
        return

    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            _mismatch(mismatches, path, f"expected number, found {_describe(actual)}")
            return
        if math.isnan(expected) and math.isnan(float(actual)):
            return
        if not math.isfinite(expected) or not math.isfinite(float(actual)):
            if expected != actual:
                _mismatch(mismatches, path, f"expected {expected!r}, found {actual!r}")
            return
        if abs(float(expected) - float(actual)) > tolerance:
            _mismatch(
                mismatches,
                path,
                f"expected {expected!r}, found {actual!r} "
                f"(tolerance {tolerance:g})",
            )
        return

    if expected is None:
        if actual is not None:
            _mismatch(mismatches, path, f"expected null, found {_describe(actual)}")
        return

    if isinstance(expected, str):
        if not isinstance(actual, str) or expected != actual:
            _mismatch(mismatches, path, f"expected {expected!r}, found {_describe(actual)}")
        return

    if expected != actual:
        _mismatch(mismatches, path, f"expected {_describe(expected)}, found {_describe(actual)}")


def _compare_keys(
    expected: Any,
    actual: Any,
    path: str,
    mismatches: list[str],
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping):
            return
        for key, value in expected.items():
            if _excluded(str(key)):
                continue
            child = f"{path}.{key}"
            if key not in actual:
                _mismatch(mismatches, child, "released field is missing from replay")
                continue
            _compare_keys(value, actual[key], child, mismatches)
        for key in actual:
            if _excluded(str(key)) or key in expected:
                continue
            _mismatch(
                mismatches,
                f"{path}.{key}",
                "replay produced a field absent from the release",
            )
        return
    if isinstance(expected, list) and isinstance(actual, list):
        for index in range(min(len(expected), len(actual))):
            _compare_keys(
                expected[index],
                actual[index],
                f"{path}[{index}]",
                mismatches,
            )


def _compare_comparisons(
    expected: Any,
    actual: Any,
    path: str,
    mismatches: list[str],
    tolerance: float,
) -> None:
    if not isinstance(expected, list):
        _mismatch(mismatches, path, "expected a list")
        return
    if not isinstance(actual, list):
        _mismatch(mismatches, path, "replay did not produce a list")
        return

    expected_by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    actual_by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for label, entries, target in (
        ("released", expected, expected_by_key),
        ("replay", actual, actual_by_key),
    ):
        for entry in entries:
            if not isinstance(entry, Mapping):
                _mismatch(
                    mismatches,
                    path,
                    f"{label} entry is not an object",
                )
                continue
            key = _comparison_key(entry)
            if key in target:
                _mismatch(
                    mismatches,
                    path,
                    f"duplicate {label} comparison {key}",
                )
                continue
            target[key] = entry

    missing = sorted(set(expected_by_key) - set(actual_by_key))
    extra = sorted(set(actual_by_key) - set(expected_by_key))
    for key in missing:
        _mismatch(
            mismatches,
            f"{path}[{key}]",
            "released comparison is missing from replay",
        )
    for key in extra:
        _mismatch(
            mismatches,
            f"{path}[{key}]",
            "replay produced a comparison absent from the release",
        )
    for key in sorted(set(expected_by_key) & set(actual_by_key)):
        _compare_value(
            expected_by_key[key],
            actual_by_key[key],
            f"{path}[{key}]",
            mismatches,
            tolerance,
        )


def _normalized_condition_lists(
    summaries: Mapping[str, Any],
) -> dict[str, Any]:
    """Sort membership-only condition lists so their order is not compared."""
    normalized = dict(summaries)
    for key in ORDER_INSENSITIVE_LIST_KEYS:
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = sorted(value, key=repr)
    return normalized


def verify_report(
    released_summaries: Mapping[str, Any],
    released_comparisons: Mapping[str, Any],
    regenerated_summaries: Mapping[str, Any],
    regenerated_comparisons: Mapping[str, Any],
    tolerance: float = TOLERANCE,
) -> None:
    """Raise ``ReplayMismatch`` when released analytical fields differ."""
    mismatches: list[str] = []
    _compare_value(
        _normalized_condition_lists(released_summaries),
        _normalized_condition_lists(regenerated_summaries),
        "summaries",
        mismatches,
        tolerance,
    )
    if "summaries" in released_summaries:
        _compare_keys(
            released_summaries["summaries"],
            regenerated_summaries.get("summaries"),
            "summaries.summaries",
            mismatches,
        )
    regenerated_grouped = regenerated_summaries.get("grouped_summaries", {})
    for group, value in released_summaries.get("grouped_summaries", {}).items():
        if group in regenerated_grouped:
            _compare_keys(
                value,
                regenerated_grouped[group],
                f"summaries.grouped_summaries.{group}",
                mismatches,
            )
    if "per_transform" in released_summaries:
        _compare_keys(
            released_summaries["per_transform"],
            regenerated_summaries.get("per_transform"),
            "summaries.per_transform",
            mismatches,
        )
    list_keys = {"comparisons", "secondary_compositions"}
    expected_top_level = {
        key: value
        for key, value in released_comparisons.items()
        if key not in list_keys
    }
    actual_top_level = {
        key: value
        for key, value in regenerated_comparisons.items()
        if key not in list_keys
    }
    _compare_value(
        expected_top_level,
        actual_top_level,
        "comparisons",
        mismatches,
        tolerance,
    )
    _compare_comparisons(
        released_comparisons.get("comparisons"),
        regenerated_comparisons.get("comparisons"),
        "comparisons.comparisons",
        mismatches,
        tolerance,
    )
    if "secondary_compositions" in released_comparisons:
        _compare_comparisons(
            released_comparisons.get("secondary_compositions"),
            regenerated_comparisons.get("secondary_compositions"),
            "comparisons.secondary_compositions",
            mismatches,
            tolerance,
        )
    if mismatches:
        shown = mismatches[:25]
        suffix = (
            f"\n... and {len(mismatches) - len(shown)} more mismatches"
            if len(mismatches) > len(shown)
            else ""
        )
        raise ReplayMismatch(
            "released report does not match the independent replay:\n"
            + "\n".join(shown)
            + suffix
        )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run_replay(
    source: str | Path,
    output: str | Path,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    """Replay one released final report into a fresh output directory."""
    output_path = Path(output)
    if output_path.exists():
        raise ReplayError(f"output directory already exists: {output_path}")

    loaded = load_source(source)
    summaries = loaded["summaries"]
    comparisons = loaded["comparisons"]
    selection = loaded["selection"]
    condition_index = _condition_index(summaries)
    bootstrap_samples, bootstrap_seed = _resolve_bootstrap(
        comparisons, summaries, selection
    )
    model_id = _model_id(summaries, selection)
    seeds = _fit_seeds(condition_index, selection)

    regenerated = recompute(
        loaded["predictions"],
        condition_index=condition_index,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        model_id=model_id,
    )
    summaries_payload, comparisons_payload = build_payloads(
        summaries,
        comparisons,
        regenerated,
        bootstrap_samples,
        bootstrap_seed,
    )
    verify_report(
        summaries,
        comparisons,
        summaries_payload,
        comparisons_payload,
        tolerance,
    )

    verification = {
        "status": "ok",
        "source": str(Path(source)),
        "tolerance": tolerance,
        "prediction_rows": len(loaded["predictions"]),
        "conditions": regenerated["conditions"],
        "fit_seeds": seeds,
        "splits": regenerated["splits"],
        "task_families": regenerated["task_families"],
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed},
        "condition_summaries": len(regenerated["summaries"]),
        "comparisons": len(regenerated["comparisons"]),
        "secondary_comparisons": len(regenerated["secondary_compositions"]),
        "checks": [
            "condition_summaries",
            "grouped_summaries",
            "per_transform",
            "primary_comparisons",
            "secondary_compositions",
            "bootstrap_metadata",
        ],
    }

    output_path.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(output_path / RELEASED_SUMMARIES, summaries_payload)
    _write_json_exclusive(output_path / RELEASED_COMPARISONS, comparisons_payload)
    _write_json_exclusive(output_path / "verification.json", verification)
    return verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independently rerun the frozen final report analysis from "
            "released predictions, without model weights."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "released report directory containing predictions.jsonl.gz, "
            "summaries.json, comparisons.json, and optional selection.json"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="fresh output directory for regenerated summaries and verification",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        verification = run_replay(args.source, args.output)
    except ReplayError as error:
        print(f"replay failed: {error}", file=sys.stderr)
        return 1
    print(
        "replay verified "
        f"{verification['condition_summaries']} condition summaries and "
        f"{verification['comparisons']} comparisons -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
