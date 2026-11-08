"""Fixed four-label mean-drift correction for RTN W4 logits.

This is the secondary postprocessing diagnostic described in
``docs/calibration-plan.md``. It fits four displayed-label offsets from
calibration original/option-permutation predictions only and applies them to
raw W4 logits at evaluation. It never uses oracle answers or prediction
correctness as a fitting target.
"""

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .data.oracles import canonical_answer
from .data.schema import OPTION_LABELS

CONDITION = "rtn_w4_logit_bias"
FIT_FORMAT = "eq_quant_w4_logit_bias_fit_v1"
CALIBRATION_SPLIT = "calibration"
FIT_TRANSFORMS = ("original", "option_permutation")
BUNDLE_FILENAMES = {
    "fp": "predictions_fp.jsonl",
    "w4": "predictions_w4.jsonl",
    "w8": "predictions_w8.jsonl",
    "w3": "predictions_w3.jsonl",
}
FIT_METADATA_FIELDS = (
    "base_id",
    "split",
    "task_family",
    "transform",
    "option_order",
    "correct_option_index",
    "labels",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_no_overwrite(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_no_overwrite(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _row_labels(row: dict[str, Any]) -> list[str]:
    labels = row.get("labels")
    if tuple(labels or ()) != OPTION_LABELS:
        raise ValueError(
            f"record {row.get('record_id')!r} must have labels {list(OPTION_LABELS)}"
        )
    return list(labels)


def _logit_vector(row: dict[str, Any]) -> list[float]:
    values = row.get("option_logits")
    if not isinstance(values, list) or len(values) != len(OPTION_LABELS):
        raise ValueError(
            f"record {row.get('record_id')!r} must have four option_logits"
        )
    logits: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"record {row.get('record_id')!r} has a non-finite option logit"
            )
        logits.append(float(value))
    return logits


def _option_order(row: dict[str, Any]) -> list[int]:
    option_order = row.get("option_order")
    if not isinstance(option_order, list) or sorted(option_order) != [0, 1, 2, 3]:
        raise ValueError(
            f"record {row.get('record_id')!r} has an invalid option_order"
        )
    return [int(index) for index in option_order]


def _index_unique(
    rows: list[dict[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} prediction is missing record_id")
        if record_id in indexed:
            raise ValueError(f"{label} predictions duplicate record_id {record_id}")
        indexed[record_id] = row
    return indexed


def _validate_calibration_only(
    rows: list[dict[str, Any]],
    label: str,
) -> None:
    if not rows:
        raise ValueError(f"{label} predictions are empty")
    splits = sorted({row.get("split") for row in rows})
    if splits != [CALIBRATION_SPLIT]:
        raise ValueError(
            f"{label} fit accepts only {CALIBRATION_SPLIT!r} records; "
            f"found splits {splits}"
        )


def _select_fit_rows(
    rows: list[dict[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed = _index_unique(rows, label)
    selected = {
        record_id: row
        for record_id, row in indexed.items()
        if row.get("transform") in FIT_TRANSFORMS
    }
    if not selected:
        raise ValueError(
            f"{label} has no original/option_permutation calibration records"
        )
    return selected


def _validate_selected_metadata(
    rows: dict[str, dict[str, Any]],
    label: str,
) -> None:
    for record_id, row in rows.items():
        missing = [field for field in FIT_METADATA_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{label} record {record_id} is missing {missing}")
        _row_labels(row)


def _metadata(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in FIT_METADATA_FIELDS)


def _match_conditions(
    fp_rows: dict[str, dict[str, Any]],
    w4_rows: dict[str, dict[str, Any]],
) -> list[str]:
    fp_ids = set(fp_rows)
    w4_ids = set(w4_rows)
    if fp_ids != w4_ids:
        missing_w4 = sorted(fp_ids - w4_ids)
        missing_fp = sorted(w4_ids - fp_ids)
        raise ValueError(
            "fp and w4 calibration support differs: "
            f"missing_w4={missing_w4}, missing_fp={missing_fp}"
        )

    for record_id in sorted(fp_ids):
        if _metadata(fp_rows[record_id]) != _metadata(w4_rows[record_id]):
            raise ValueError(
                f"record {record_id} metadata differs between fp and w4"
            )
    return sorted(fp_ids)


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def fit_offsets(
    fp_rows: list[dict[str, Any]],
    w4_rows: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    fp_path: str | Path | None = None,
    w4_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit displayed-label offsets from calibration predictions only."""
    _validate_calibration_only(fp_rows, "fp")
    _validate_calibration_only(w4_rows, "w4")

    fp_selected = _select_fit_rows(fp_rows, "fp")
    w4_selected = _select_fit_rows(w4_rows, "w4")
    _validate_selected_metadata(fp_selected, "fp")
    _validate_selected_metadata(w4_selected, "w4")
    record_ids = _match_conditions(fp_selected, w4_selected)

    offsets = {label: 0.0 for label in OPTION_LABELS}
    for record_id in record_ids:
        fp_logits = _logit_vector(fp_selected[record_id])
        w4_logits = _logit_vector(w4_selected[record_id])
        differences = [
            fp_logit - w4_logit
            for fp_logit, w4_logit in zip(fp_logits, w4_logits, strict=True)
        ]
        row_mean = sum(differences) / len(differences)
        for label, difference in zip(OPTION_LABELS, differences, strict=True):
            offsets[label] += difference - row_mean

    for label in OPTION_LABELS:
        offsets[label] /= len(record_ids)

    selected_rows = [fp_selected[record_id] for record_id in record_ids]
    zero_sum_residual = float(sum(offsets.values()))
    model_provenance = None
    if config is not None:
        model_provenance = config.get("model", config)

    return {
        "format": FIT_FORMAT,
        "state": "completed",
        "condition": CONDITION,
        "method": "four_displayed_label_mean_drift_correction",
        "offsets": {label: offsets[label] for label in OPTION_LABELS},
        "zero_sum_residual": zero_sum_residual,
        "fit": {
            "split": CALIBRATION_SPLIT,
            "transforms": list(FIT_TRANSFORMS),
            "record_count": len(record_ids),
            "base_count": len({row["base_id"] for row in selected_rows}),
            "by_transform": _counts(selected_rows, "transform"),
            "by_task_family": _counts(selected_rows, "task_family"),
        },
        "sources": {
            "fp": {
                "path": None if fp_path is None else str(fp_path),
                "rows": len(fp_rows),
                "selected_rows": len(fp_selected),
            },
            "w4": {
                "path": None if w4_path is None else str(w4_path),
                "rows": len(w4_rows),
                "selected_rows": len(w4_selected),
            },
        },
        "model_provenance": model_provenance,
        "targets": {
            "uses_oracle_labels": False,
            "uses_prediction_correctness": False,
            "uses_development_or_final_outcomes": False,
        },
        "created_at": _now(),
    }


def _offsets_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    if payload.get("format") != FIT_FORMAT:
        raise ValueError(
            f"unsupported logit-bias fit format: {payload.get('format')!r}"
        )
    offsets = payload.get("offsets")
    if not isinstance(offsets, dict) or tuple(sorted(offsets)) != tuple(
        sorted(OPTION_LABELS)
    ):
        raise ValueError(
            f"offsets must define exactly the displayed labels {list(OPTION_LABELS)}"
        )
    result: dict[str, float] = {}
    for label in OPTION_LABELS:
        value = offsets[label]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"offset for label {label!r} is not finite")
        result[label] = float(value)
    return result


def _canonicalize_probabilities(
    display_probabilities: list[float],
    option_order: list[int],
) -> list[float]:
    canonical = [0.0] * len(display_probabilities)
    for display_index, canonical_index in enumerate(option_order):
        canonical[canonical_index] = display_probabilities[display_index]
    return canonical


def apply_offsets(
    w4_rows: list[dict[str, Any]],
    fit_payload: dict[str, Any],
    source_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Add fixed FP32 displayed-label offsets to raw W4 option logits."""
    offsets = _offsets_from_payload(fit_payload)
    source_bundle = None if source_path is None else str(source_path)
    fit_summary = {
        "format": fit_payload.get("format"),
        "fit_split": fit_payload.get("fit", {}).get("split"),
        "fit_transforms": fit_payload.get("fit", {}).get("transforms"),
        "fit_records": fit_payload.get("fit", {}).get("record_count"),
        "method": fit_payload.get("method"),
    }

    adjusted_rows: list[dict[str, Any]] = []
    for row in w4_rows:
        labels = _row_labels(row)
        raw_logits = _logit_vector(row)
        option_order = _option_order(row)

        logits = torch.tensor(raw_logits, dtype=torch.float32)
        offset_tensor = torch.tensor(
            [offsets[label] for label in labels], dtype=torch.float32
        )
        shifted_logits = logits + offset_tensor
        probabilities = torch.softmax(shifted_logits, dim=-1)

        prediction_display = int(torch.argmax(shifted_logits).item())
        canonical_prediction = option_order[prediction_display]
        display_probabilities = probabilities.tolist()
        canonical_probabilities = _canonicalize_probabilities(
            display_probabilities, option_order
        )

        correct_canonical = canonical_answer(
            row["task_family"],
            row["semantic"]["structure"],
            row["semantic"]["canonical_options"],
        )
        expected_display = option_order.index(correct_canonical)
        if int(row["correct_option_index"]) != expected_display:
            raise ValueError(
                f"record {row['record_id']} failed oracle verification"
            )

        output = dict(row)
        output.update(
            {
                "condition": CONDITION,
                "option_logits": shifted_logits.tolist(),
                "option_probabilities": display_probabilities,
                "canonical_probabilities": canonical_probabilities,
                "prediction_display": prediction_display,
                "canonical_prediction": canonical_prediction,
                "correct": canonical_prediction == correct_canonical,
                "raw_option_logits": raw_logits,
                "source_condition": row.get("condition"),
                "bias_offsets": {label: offsets[label] for label in labels},
                "bias_fit": fit_summary,
                "bias_source_predictions": source_bundle,
            }
        )
        adjusted_rows.append(output)
    return adjusted_rows


def bundle_paths(bundle: str | Path) -> dict[str, Path]:
    root = Path(bundle)
    if not root.is_dir():
        raise FileNotFoundError(f"prediction bundle is not a directory: {root}")
    paths = {name: root / filename for name, filename in BUNDLE_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "prediction bundle is missing FP/W4/W8/W3 files: " f"{missing}"
        )
    return paths


def load_bundle(bundle: str | Path) -> dict[str, list[dict[str, Any]]]:
    paths = bundle_paths(bundle)
    return {name: load_jsonl(path) for name, path in paths.items()}


def _fit_command(args: argparse.Namespace) -> None:
    fp_rows = load_jsonl(args.fp)
    w4_rows = load_jsonl(args.w4)
    config = None if args.config is None else load_json(args.config)
    payload = fit_offsets(
        fp_rows,
        w4_rows,
        config=config,
        fp_path=args.fp,
        w4_path=args.w4,
    )
    _write_json_no_overwrite(args.output, payload)
    print(
        json.dumps(
            {
                "condition": payload["condition"],
                "records": payload["fit"]["record_count"],
                "offsets": payload["offsets"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _apply_command(args: argparse.Namespace) -> None:
    fit_payload = load_json(args.offsets)
    paths = bundle_paths(args.bundle)
    w4_rows = load_jsonl(paths["w4"])
    rows = apply_offsets(w4_rows, fit_payload, source_path=paths["w4"])
    _write_jsonl_no_overwrite(args.output, rows)
    print(
        json.dumps(
            {
                "condition": CONDITION,
                "rows": len(rows),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eq_quant.bias_control",
        description=(
            "Fit or apply the fixed four-label mean-drift correction to RTN W4 "
            "option logits."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser(
        "fit",
        help="fit displayed-label offsets from calibration FP and W4 predictions",
    )
    fit_parser.add_argument("--fp", required=True, help="FP calibration predictions JSONL")
    fit_parser.add_argument("--w4", required=True, help="W4 calibration predictions JSONL")
    fit_parser.add_argument("--output", required=True, help="new fit JSON path")
    fit_parser.add_argument(
        "--config",
        default=None,
        help="optional config JSON supplying model provenance",
    )
    fit_parser.set_defaults(handler=_fit_command)

    apply_parser = subparsers.add_parser(
        "apply",
        help="add fixed offsets to W4 rows from an FP/W4/W8/W3 bundle",
    )
    apply_parser.add_argument("--offsets", required=True, help="fit JSON path")
    apply_parser.add_argument(
        "--bundle",
        required=True,
        help="directory containing predictions_fp/w4/w8/w3.jsonl",
    )
    apply_parser.add_argument("--output", required=True, help="new predictions JSONL path")
    apply_parser.set_defaults(handler=_apply_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
