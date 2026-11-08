"""Explicit phase runner for the frozen calibration study.

This module orchestrates the existing data, calibration, evaluation, and metric
APIs. It does not reimplement training, scoring, or aggregation. The CLI is

    python -m eq_quant.study --config CONFIG --root ROOT --phase PHASE

where PHASE is one of ``develop``, ``replicate``, ``diagnostic``, ``final``,
or ``report``.
"""

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analyze.stats import load_predictions
from .bias_control import apply_offsets, fit_offsets
from .calibrate.evaluate import (
    MANIFEST_FILENAME as EVALUATION_MANIFEST,
    PREDICTIONS_FILENAME,
    run_calibrated_evaluation,
)
from .calibrate.records import select_calibration_pairs
from .data.oracles import canonical_answer
from .data.schema import COMPOSED_TRANSFORMS, OPTION_LABELS
from .calibrate.io_helpers import write_json, write_jsonl
from .calibrate.teacher import (
    CACHE_FILENAME,
    MANIFEST_FILENAME as CACHE_MANIFEST,
    cache_teacher,
)
from .calibrate.train import (
    MANIFEST_FILENAME as FIT_MANIFEST,
    SCALES_FILENAME,
    run_calibration,
)
from .config import load_config
from .data.build import build_dataset, write_dataset
from .evaluate.runner import load_records, run_evaluation
from .study_metrics import (
    HELD_OUT_TRANSFORMS,
    ORIGINAL_TRANSFORM,
    compare_conditions,
    recompute_correct,
    select_lambda,
    summarize_condition,
    summarize_conditions,
)

PHASES = ("develop", "replicate", "diagnostic", "final", "report")
DEVELOPMENT_SEED = 17
CONTROL_OBJECTIVES = ("original_kd", "augmented_kd")
REGULARIZED_OBJECTIVES = ("paired", "random_pair", "kernel")
SELECTION_FORMAT = "eq_quant_study_selection_v1"
REPORT_FORMAT = "eq_quant_study_report_v1"
METADATA_FORMAT = "eq_quant_study_artifact_v1"
CONSTANT_DISPLAY_A_CONDITION = "constant_display_A"
UNTRAINED_RULE_FORMAT = "eq_quant_study_untrained_rule_v1"
UNTRAINED_RULE_MANIFEST = "untrained_rule.json"
DIAGNOSTIC_FORMAT = "eq_quant_study_diagnostic_v1"
DIAGNOSTIC_RECORDS_FILENAME = "calibration_inputs.jsonl"
DIAGNOSTIC_METADATA_FILENAME = "study_diagnostic.json"
DIAGNOSTIC_OFFSETS_FILENAME = "logit_bias_offsets.json"
DIAGNOSTIC_EVALUATION_DIR = "evaluation"
DIAGNOSTIC_RECORD_COUNT = 768
DIAGNOSTIC_BITS = (4,)
RTN_BIAS_CONDITION = "rtn_w4_logit_bias"
RTN_BIAS_OBJECTIVE = "mean_logit_drift"
POSTPROCESSING_FORMAT = "eq_quant_study_secondary_postprocessing_v1"
POSTPROCESSING_MANIFEST = "postprocessing_manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(phase: str, message: str) -> None:
    print(f"[{phase}] {message}", flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_no_overwrite(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_no_overwrite(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _append_journal(root: Path, entry: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _git(
    arguments: list[str],
    repository: Path,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None


def _provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    revision_result = _git(["rev-parse", "HEAD"], repository)
    status_result = _git(["status", "--porcelain"], repository)
    revision = None
    if revision_result is not None and revision_result.returncode == 0:
        revision = revision_result.stdout.strip() or None
    dirty_paths: list[str] = []
    if status_result is not None and status_result.returncode == 0:
        dirty_paths = sorted(
            line[3:].strip()
            for line in status_result.stdout.splitlines()
            if line.strip()
        )
    return {
        "git_revision": revision,
        "dirty_paths": dirty_paths,
    }


def _same_lambda(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return float(left) == float(right)


def _lambda_tag(value: float) -> str:
    return f"{float(value):g}"


def _version(objective: str, lambda_value: float | None) -> str:
    if lambda_value is None:
        return objective
    return f"{objective}-lam{_lambda_tag(lambda_value)}"


def _fit_dir(root: Path, objective: str, lambda_value: float | None, seed: int) -> Path:
    return root / "fits" / _version(objective, lambda_value) / f"seed-{seed}"


def _development_dir(
    root: Path, objective: str, lambda_value: float | None, seed: int
) -> Path:
    return root / "development" / _version(objective, lambda_value) / f"seed-{seed}"


def _final_dir(
    root: Path, objective: str, lambda_value: float | None, seed: int
) -> Path:
    return root / "results" / "final" / _version(objective, lambda_value) / f"seed-{seed}"


def _development_condition(objective: str, lambda_value: float | None, seed: int) -> str:
    return f"{_version(objective, lambda_value)}_s{seed}"


def _final_condition(objective: str, seed: int) -> str:
    return f"{objective}_s{seed}"


def _require_files(directory: Path, filenames: list[str], label: str) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {directory}")
    missing = [name for name in filenames if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"{label} is incomplete at {directory}: missing {missing}")


def _check_config(config: dict[str, Any], recorded: Any, label: str) -> None:
    if recorded != config:
        raise ValueError(f"{label} was produced from a different config")


def _check_model(config: dict[str, Any], recorded: Any, label: str) -> None:
    if not isinstance(recorded, dict):
        raise ValueError(f"{label} lacks recorded model provenance")
    if recorded.get("id") != config["model"]["id"]:
        raise ValueError(f"{label} was produced from a different model id")
    if recorded.get("revision") != config["model"]["revision"]:
        raise ValueError(f"{label} was produced from a different model revision")


def _ensure_dataset(
    config: dict[str, Any],
    output_dir: Path,
    split_names: list[str],
    provenance: dict[str, Any],
    phase: str,
) -> Path:
    metadata_path = output_dir / "study_data.json"
    records_path = output_dir / "records.jsonl"
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        _require_files(
            output_dir,
            ["records.jsonl", "manifest.json", "study_data.json"],
            "dataset",
        )
        metadata = _read_json(metadata_path)
        if metadata.get("format") != METADATA_FORMAT:
            raise ValueError(f"dataset metadata format is unsupported: {metadata_path}")
        _check_config(config, metadata.get("config"), "dataset")
        if metadata.get("splits") != split_names:
            raise ValueError(f"dataset was generated for different splits: {output_dir}")
        _log(phase, f"reuse dataset {output_dir}")
        return records_path

    _log(phase, f"generate dataset {output_dir} splits={split_names}")
    dataset = build_dataset(config, split_names=split_names)
    write_dataset(dataset, output_dir)
    write_json(
        metadata_path,
        {
            "format": METADATA_FORMAT,
            "kind": "dataset",
            "splits": split_names,
            "config": config,
            "source": provenance,
            "created_at": _now(),
        },
    )
    return records_path


def _ensure_teacher_cache(
    config: dict[str, Any],
    records_path: Path,
    output_dir: Path,
    provenance: dict[str, Any],
    phase: str,
) -> Path:
    metadata_path = output_dir / "study_cache.json"
    if output_dir.exists():
        _require_files(
            output_dir,
            [CACHE_FILENAME, CACHE_MANIFEST, "study_cache.json"],
            "teacher cache",
        )
        manifest = _read_json(output_dir / CACHE_MANIFEST)
        metadata = _read_json(metadata_path)
        if metadata.get("format") != METADATA_FORMAT:
            raise ValueError(f"teacher cache metadata format is unsupported: {metadata_path}")
        _check_config(config, metadata.get("config"), "teacher cache")
        _check_model(config, manifest.get("model"), "teacher cache")
        if str(metadata.get("records_path")) != str(records_path):
            raise ValueError(f"teacher cache was built from different records: {output_dir}")
        if manifest.get("state") != "completed":
            raise RuntimeError(
                f"teacher cache at {output_dir} is not completed: "
                f"state={manifest.get('state')!r}"
            )
        _log(phase, f"reuse teacher cache {output_dir}")
        return output_dir / CACHE_FILENAME

    _log(phase, f"cache teacher logits {output_dir}")
    cache_teacher(
        config=config,
        records_path=records_path,
        output_dir=output_dir,
        overwrite=False,
    )
    write_json(
        metadata_path,
        {
            "format": METADATA_FORMAT,
            "kind": "teacher_cache",
            "config": config,
            "records_path": str(records_path),
            "source": provenance,
            "created_at": _now(),
        },
    )
    return output_dir / CACHE_FILENAME


def _load_fit(
    config: dict[str, Any],
    fit_dir: Path,
    objective: str,
    lambda_value: float | None,
    seed: int,
) -> dict[str, Any]:
    _require_files(
        fit_dir,
        [FIT_MANIFEST, SCALES_FILENAME],
        "calibration fit",
    )
    manifest = _read_json(fit_dir / FIT_MANIFEST)
    if manifest.get("state") != "completed":
        raise RuntimeError(
            f"calibration fit at {fit_dir} is not completed: "
            f"state={manifest.get('state')!r}"
        )
    if manifest.get("objective") != objective:
        raise ValueError(f"calibration fit objective mismatch at {fit_dir}")
    if not _same_lambda(manifest.get("lambda"), lambda_value):
        raise ValueError(f"calibration fit lambda mismatch at {fit_dir}")
    if int(manifest.get("fit_seed", -1)) != int(seed):
        raise ValueError(f"calibration fit seed mismatch at {fit_dir}")
    _check_config(config, manifest.get("config"), "calibration fit")
    _check_model(config, manifest.get("model"), "calibration fit")
    return manifest


def _ensure_fit(
    config: dict[str, Any],
    records_path: Path,
    teacher_cache_path: Path,
    output_dir: Path,
    objective: str,
    lambda_value: float | None,
    seed: int,
    phase: str,
) -> dict[str, Any]:
    if output_dir.exists():
        _log(phase, f"reuse fit {output_dir}")
        return _load_fit(config, output_dir, objective, lambda_value, seed)

    _log(
        phase,
        f"fit objective={objective} lambda={lambda_value} seed={seed} -> {output_dir}",
    )
    return run_calibration(
        config=config,
        records_path=records_path,
        teacher_cache_path=teacher_cache_path,
        output_dir=output_dir,
        objective=objective,
        lambda_value=lambda_value,
        seed=seed,
        overwrite=False,
    )


def _load_calibrated_evaluation(
    config: dict[str, Any],
    evaluation_dir: Path,
    records_path: Path,
    scales_path: Path,
    objective: str,
    lambda_value: float | None,
    seed: int,
    condition: str,
) -> dict[str, Any]:
    _require_files(
        evaluation_dir,
        [EVALUATION_MANIFEST, PREDICTIONS_FILENAME, "study_evaluation.json"],
        "calibrated evaluation",
    )
    manifest = _read_json(evaluation_dir / EVALUATION_MANIFEST)
    metadata = _read_json(evaluation_dir / "study_evaluation.json")
    if metadata.get("format") != METADATA_FORMAT:
        raise ValueError(
            f"evaluation metadata format is unsupported: {evaluation_dir}"
        )
    _check_config(config, metadata.get("config"), "calibrated evaluation")
    _check_model(config, manifest.get("model"), "calibrated evaluation")
    if manifest.get("state") != "completed":
        raise RuntimeError(
            f"calibrated evaluation at {evaluation_dir} is not completed: "
            f"state={manifest.get('state')!r}"
        )
    if manifest.get("condition") != condition:
        raise ValueError(f"calibrated evaluation condition mismatch at {evaluation_dir}")
    if metadata.get("objective") != objective:
        raise ValueError(f"calibrated evaluation objective mismatch at {evaluation_dir}")
    if not _same_lambda(metadata.get("lambda"), lambda_value):
        raise ValueError(f"calibrated evaluation lambda mismatch at {evaluation_dir}")
    if int(metadata.get("fit_seed", -1)) != int(seed):
        raise ValueError(f"calibrated evaluation seed mismatch at {evaluation_dir}")
    if str(metadata.get("records_path")) != str(records_path):
        raise ValueError(f"calibrated evaluation records mismatch at {evaluation_dir}")
    if str(metadata.get("scales_path")) != str(scales_path):
        raise ValueError(f"calibrated evaluation scales mismatch at {evaluation_dir}")
    return metadata


def _ensure_calibrated_evaluation(
    config: dict[str, Any],
    records_path: Path,
    scales_path: Path,
    output_dir: Path,
    objective: str,
    lambda_value: float | None,
    seed: int,
    condition: str,
    provenance: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    if output_dir.exists():
        _log(phase, f"reuse evaluation {output_dir}")
        return _load_calibrated_evaluation(
            config,
            output_dir,
            records_path,
            scales_path,
            objective,
            lambda_value,
            seed,
            condition,
        )

    _log(
        phase,
        f"evaluate condition={condition} records={records_path} -> {output_dir}",
    )
    manifest = run_calibrated_evaluation(
        config=config,
        records_path=records_path,
        scales_path=scales_path,
        output_dir=output_dir,
        condition=condition,
        overwrite=False,
    )
    write_json(
        output_dir / "study_evaluation.json",
        {
            "format": METADATA_FORMAT,
            "kind": "calibrated_evaluation",
            "config": config,
            "records_path": str(records_path),
            "scales_path": str(scales_path),
            "objective": objective,
            "lambda": lambda_value,
            "fit_seed": seed,
            "condition": condition,
            "source": provenance,
            "created_at": _now(),
        },
    )
    return manifest


def _load_baseline_evaluation(
    config: dict[str, Any],
    evaluation_dir: Path,
    records_path: Path,
) -> dict[str, Any]:
    _require_files(
        evaluation_dir,
        ["run.json", "predictions.jsonl", "study_evaluation.json"],
        "baseline evaluation",
    )
    manifest = _read_json(evaluation_dir / "run.json")
    metadata = _read_json(evaluation_dir / "study_evaluation.json")
    if metadata.get("format") != METADATA_FORMAT:
        raise ValueError(
            f"baseline metadata format is unsupported: {evaluation_dir}"
        )
    _check_config(config, metadata.get("config"), "baseline evaluation")
    _check_model(config, manifest.get("model"), "baseline evaluation")
    if str(metadata.get("records_path")) != str(records_path):
        raise ValueError(f"baseline evaluation records mismatch at {evaluation_dir}")
    if manifest.get("state") != "completed":
        raise RuntimeError(
            f"baseline evaluation at {evaluation_dir} is not completed: "
            f"state={manifest.get('state')!r}"
        )
    if "fp" not in list(manifest.get("completed_conditions", [])):
        raise RuntimeError(f"baseline evaluation at {evaluation_dir} lacks fp")
    return manifest


def _ensure_baseline_evaluation(
    config: dict[str, Any],
    records_path: Path,
    output_dir: Path,
    provenance: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    if output_dir.exists():
        _log(phase, f"reuse baseline evaluation {output_dir}")
        return _load_baseline_evaluation(config, output_dir, records_path)

    _log(phase, f"evaluate FP/RTN baseline records={records_path} -> {output_dir}")
    manifest = run_evaluation(
        config=config,
        records_path=records_path,
        output_dir=output_dir,
        generation_subset=int(config.get("generation", {}).get("subset_size", 0)),
        overwrite=False,
    )
    write_json(
        output_dir / "study_evaluation.json",
        {
            "format": METADATA_FORMAT,
            "kind": "baseline_evaluation",
            "config": config,
            "records_path": str(records_path),
            "source": provenance,
            "created_at": _now(),
        },
    )
    return manifest


def _diagnostic_dir(root: Path) -> Path:
    return root / "diagnostic"


def _diagnostic_records_path(root: Path) -> Path:
    return _diagnostic_dir(root) / DIAGNOSTIC_RECORDS_FILENAME


def _diagnostic_metadata_path(root: Path) -> Path:
    return _diagnostic_dir(root) / DIAGNOSTIC_METADATA_FILENAME


def _diagnostic_offsets_path(root: Path) -> Path:
    return _diagnostic_dir(root) / DIAGNOSTIC_OFFSETS_FILENAME


def _diagnostic_evaluation_dir(root: Path) -> Path:
    return _diagnostic_dir(root) / DIAGNOSTIC_EVALUATION_DIR


def _rtn_bias_dir(root: Path) -> Path:
    return root / "results" / "final" / RTN_BIAS_CONDITION


def _constant_display_a_dir(root: Path) -> Path:
    return root / "results" / "final" / "constant_display_A"


def _constant_display_a_manifest(
    config: dict[str, Any],
    records_path: Path,
    row_count: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": UNTRAINED_RULE_FORMAT,
        "kind": "constant_display_A",
        "state": "completed",
        "condition": CONSTANT_DISPLAY_A_CONDITION,
        "rule": "constant_display_A",
        "decision_rule": {
            "prediction_display": 0,
            "canonical_prediction": "option_order[0]",
        },
        "untrained_rule": True,
        "uses_model": False,
        "uses_fit": False,
        "uses_teacher": False,
        "replaces_paired_vs_augmented": False,
        "config": config,
        "model": config["model"],
        "records_path": str(records_path),
        "record_count": row_count,
        "source": provenance,
        "created_at": _now(),
    }


def _constant_display_a_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the fixed displayed-A rule to records without model outputs."""
    rows: list[dict[str, Any]] = []
    for record in records:
        option_order = [int(index) for index in record["option_order"]]
        gold = canonical_answer(
            record["task_family"],
            record["semantic"]["structure"],
            record["semantic"]["canonical_options"],
        )
        expected_display = option_order.index(gold)
        if int(record["correct_option_index"]) != expected_display:
            raise ValueError(
                f"record {record['record_id']} failed oracle verification"
            )

        prediction_display = 0
        canonical_prediction = option_order[prediction_display]
        rows.append(
            {
                "record_id": record["record_id"],
                "base_id": record["base_id"],
                "split": record["split"],
                "task_family": record["task_family"],
                "transform": record["transform"],
                "condition": CONSTANT_DISPLAY_A_CONDITION,
                "semantic": record["semantic"],
                "option_order": option_order,
                "labels": list(OPTION_LABELS),
                "prediction_display": prediction_display,
                "canonical_prediction": canonical_prediction,
                "correct_option_index": int(record["correct_option_index"]),
                "correct": canonical_prediction == gold,
                "untrained_rule": True,
            }
        )
    return rows


def _load_constant_display_a_control(
    config: dict[str, Any],
    evaluation_dir: Path,
    records_path: Path,
) -> dict[str, Any]:
    _require_files(
        evaluation_dir,
        [PREDICTIONS_FILENAME, UNTRAINED_RULE_MANIFEST],
        "constant displayed-A control",
    )
    manifest = _read_json(evaluation_dir / UNTRAINED_RULE_MANIFEST)
    if manifest.get("format") != UNTRAINED_RULE_FORMAT:
        raise ValueError(
            f"unsupported untrained-rule manifest format: {evaluation_dir}"
        )
    _check_config(config, manifest.get("config"), "constant displayed-A control")
    _check_model(config, manifest.get("model"), "constant displayed-A control")
    if manifest.get("condition") != CONSTANT_DISPLAY_A_CONDITION:
        raise ValueError(
            f"constant displayed-A control condition mismatch at {evaluation_dir}"
        )
    if manifest.get("state") != "completed":
        raise RuntimeError(
            f"constant displayed-A control at {evaluation_dir} is not completed: "
            f"state={manifest.get('state')!r}"
        )
    if manifest.get("untrained_rule") is not True or manifest.get("uses_model") is not False:
        raise ValueError(
            f"constant displayed-A control at {evaluation_dir} is not untrained"
        )
    if str(manifest.get("records_path")) != str(records_path):
        raise ValueError(
            f"constant displayed-A control records mismatch at {evaluation_dir}"
        )
    return manifest


def _ensure_constant_display_a_control(
    config: dict[str, Any],
    records_path: Path,
    evaluation_dir: Path,
    provenance: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    if evaluation_dir.exists():
        if not (evaluation_dir / UNTRAINED_RULE_MANIFEST).is_file():
            raise RuntimeError(
                "constant displayed-A control directory exists without an "
                f"untrained-rule manifest; refusing to overwrite: {evaluation_dir}"
            )
        _log(phase, f"reuse constant displayed-A control {evaluation_dir}")
        return _load_constant_display_a_control(
            config, evaluation_dir, records_path
        )

    records = load_records(records_path)
    if not records:
        raise ValueError("no final records for the constant displayed-A control")
    rows = _constant_display_a_rows(records)
    _log(
        phase,
        f"write untrained constant displayed-A control rows={len(rows)} "
        f"-> {evaluation_dir}",
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(evaluation_dir / PREDICTIONS_FILENAME, rows)
    manifest = _constant_display_a_manifest(
        config, records_path, len(rows), provenance
    )
    write_json(evaluation_dir / UNTRAINED_RULE_MANIFEST, manifest)
    return manifest


def _load_rtn_bias_control(
    config: dict[str, Any],
    evaluation_dir: Path,
    records_path: Path,
    offsets_path: Path,
    offsets_payload: dict[str, Any],
) -> dict[str, Any]:
    _require_files(
        evaluation_dir,
        [PREDICTIONS_FILENAME, POSTPROCESSING_MANIFEST],
        "RTN W4 logit-bias control",
    )
    manifest = _read_json(evaluation_dir / POSTPROCESSING_MANIFEST)
    if manifest.get("format") != POSTPROCESSING_FORMAT:
        raise ValueError(
            f"unsupported postprocessing manifest format: {evaluation_dir}"
        )
    _check_config(config, manifest.get("config"), "RTN W4 logit-bias control")
    _check_model(config, manifest.get("model"), "RTN W4 logit-bias control")
    if manifest.get("state") != "completed":
        raise RuntimeError(
            f"RTN W4 logit-bias control is not completed: "
            f"state={manifest.get('state')!r}"
        )
    if manifest.get("condition") != RTN_BIAS_CONDITION:
        raise ValueError(
            f"RTN W4 logit-bias control condition mismatch at {evaluation_dir}"
        )
    if manifest.get("objective") != RTN_BIAS_OBJECTIVE:
        raise ValueError(
            f"RTN W4 logit-bias control objective mismatch at {evaluation_dir}"
        )
    if manifest.get("fit_seed") is not None:
        raise ValueError(
            f"RTN W4 logit-bias control must not use a fit seed: {evaluation_dir}"
        )
    if manifest.get("secondary_postprocessing") is not True:
        raise ValueError(
            f"RTN W4 logit-bias control must be secondary: {evaluation_dir}"
        )
    if manifest.get("replaces_paired_vs_augmented") is not False:
        raise ValueError(
            f"RTN W4 logit-bias control must not replace the primary comparison: "
            f"{evaluation_dir}"
        )
    if str(manifest.get("records_path")) != str(records_path):
        raise ValueError(
            f"RTN W4 logit-bias control records mismatch at {evaluation_dir}"
        )
    if str(manifest.get("offsets_path")) != str(offsets_path):
        raise ValueError(
            f"RTN W4 logit-bias control offsets mismatch at {evaluation_dir}"
        )
    if manifest.get("offsets") != offsets_payload.get("offsets"):
        raise ValueError(
            f"RTN W4 logit-bias control offsets differ from the diagnostic: "
            f"{evaluation_dir}"
        )
    return manifest


def _ensure_rtn_bias_control(
    config: dict[str, Any],
    records_path: Path,
    baseline_dir: Path,
    evaluation_dir: Path,
    offsets_path: Path,
    offsets_payload: dict[str, Any],
    diagnostic_metadata_path: Path,
    provenance: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    if evaluation_dir.exists():
        if not (evaluation_dir / POSTPROCESSING_MANIFEST).is_file():
            raise RuntimeError(
                "RTN W4 logit-bias control directory exists without a "
                f"postprocessing manifest; refusing to overwrite: {evaluation_dir}"
            )
        _log(phase, f"reuse RTN W4 logit-bias control {evaluation_dir}")
        return _load_rtn_bias_control(
            config,
            evaluation_dir,
            records_path,
            offsets_path,
            offsets_payload,
        )

    w4_path = baseline_dir / "predictions_w4.jsonl"
    if not w4_path.is_file():
        raise FileNotFoundError(
            f"baseline W4 predictions are missing: {w4_path}"
        )
    w4_rows = load_predictions(w4_path)
    if not w4_rows:
        raise ValueError("no baseline W4 rows for the logit-bias control")

    _log(
        phase,
        f"apply fixed W4 logit offsets rows={len(w4_rows)} -> {evaluation_dir}",
    )
    rows = apply_offsets(w4_rows, offsets_payload, source_path=w4_path)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(evaluation_dir / PREDICTIONS_FILENAME, rows)
    manifest = {
        "format": POSTPROCESSING_FORMAT,
        "kind": "secondary_mean_logit_drift",
        "state": "completed",
        "condition": RTN_BIAS_CONDITION,
        "objective": RTN_BIAS_OBJECTIVE,
        "fit_seed": None,
        "secondary_postprocessing": True,
        "replaces_paired_vs_augmented": False,
        "uses_oracle_labels": False,
        "uses_development_or_final_outcomes": False,
        "parameters": len(OPTION_LABELS),
        "config": config,
        "model": config["model"],
        "records_path": str(records_path),
        "offsets_path": str(offsets_path),
        "offsets": offsets_payload["offsets"],
        "source_predictions": str(w4_path),
        "record_count": len(rows),
        "diagnostic_metadata": str(diagnostic_metadata_path),
        "source": provenance,
        "created_at": _now(),
    }
    write_json(evaluation_dir / POSTPROCESSING_MANIFEST, manifest)
    return manifest


def _development_metrics(
    config: dict[str, Any],
    evaluation_dir: Path,
    condition: str,
) -> dict[str, Any]:
    predictions = load_predictions(evaluation_dir)
    summary = summarize_condition(predictions, condition)
    held_out = summary["endpoints"]["held_out_singles"]
    if held_out is None:
        raise ValueError(f"development evaluation {evaluation_dir} has no held-out singles")
    return {
        "overall_accuracy": summary["accuracy_macro"],
        "held_out_accuracy": held_out["accuracy_macro"],
        "held_out_jointly_correct_rate": held_out["jointly_correct_rate_macro"],
        "held_out_consistency": held_out["consistency_macro"],
        "n_bases": summary["n_bases"],
    }


def _fit_seeds(config: dict[str, Any]) -> list[int]:
    seeds = [int(value) for value in config["calibration"]["fit_seeds"]]
    if DEVELOPMENT_SEED not in seeds:
        raise ValueError(
            f"calibration.fit_seeds must contain the frozen development seed "
            f"{DEVELOPMENT_SEED}"
        )
    return seeds


def _lambda_candidates(config: dict[str, Any]) -> list[float]:
    values = [float(value) for value in config["calibration"]["lambda_candidates"]]
    if not values:
        raise ValueError("calibration.lambda_candidates must not be empty")
    return values


def _selected_entries(selection: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [
        {"objective": objective, "lambda": None}
        for objective in CONTROL_OBJECTIVES
    ]
    for objective in REGULARIZED_OBJECTIVES:
        block = selection.get("objectives", {}).get(objective)
        if not isinstance(block, dict):
            raise ValueError(f"selection lacks objective {objective!r}")
        entries.append(
            {
                "objective": objective,
                "lambda": float(block["selected_lambda"]),
            }
        )
    return entries


def _load_selection(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = root / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"frozen selection is missing: {path}")
    selection = _read_json(path)
    if selection.get("format") != SELECTION_FORMAT:
        raise ValueError(f"unsupported selection format: {path}")
    if selection.get("state") != "frozen":
        raise RuntimeError(
            f"selection at {path} is not frozen: state={selection.get('state')!r}"
        )
    _check_config(config, selection.get("config"), "frozen selection")
    _check_model(config, selection.get("model"), "frozen selection")
    return selection


def _require_selected_fits(
    config: dict[str, Any],
    root: Path,
    selection: dict[str, Any],
    seeds: list[int],
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for entry in _selected_entries(selection):
        for seed in seeds:
            fit_dir_path = _fit_dir(root, entry["objective"], entry["lambda"], seed)
            manifest = _load_fit(
                config,
                fit_dir_path,
                entry["objective"],
                entry["lambda"],
                seed,
            )
            loaded.append(
                {
                    **entry,
                    "seed": seed,
                    "path": fit_dir_path,
                    "manifest": manifest,
                }
            )
    return loaded


def _optimization_time_check(
    root: Path,
    selection: dict[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    paired_lambda = float(selection["objectives"]["paired"]["selected_lambda"])
    by_seed: dict[str, Any] = {}
    for seed in seeds:
        paired_path = _fit_dir(root, "paired", paired_lambda, seed) / FIT_MANIFEST
        augmented_path = _fit_dir(root, "augmented_kd", None, seed) / FIT_MANIFEST
        if not paired_path.is_file() or not augmented_path.is_file():
            by_seed[str(seed)] = {
                "paired_seconds": None,
                "augmented_seconds": None,
                "ratio": None,
            }
            continue
        paired_seconds = _read_json(paired_path).get("optimization_elapsed_seconds")
        augmented_seconds = _read_json(augmented_path).get("optimization_elapsed_seconds")
        ratio = None
        if paired_seconds is not None and augmented_seconds:
            ratio = float(paired_seconds) / float(augmented_seconds)
        by_seed[str(seed)] = {
            "paired_seconds": paired_seconds,
            "augmented_seconds": augmented_seconds,
            "ratio": ratio,
        }

    measured_seeds = [
        int(seed)
        for seed, value in by_seed.items()
        if value["ratio"] is not None
    ]
    required = any(
        value["ratio"] is not None and value["ratio"] > 1.10
        for value in by_seed.values()
    )
    if required:
        note = (
            "A measured paired/augmented optimization-time ratio above 1.10 "
            "requires a compute-matched augmented-KD control; none is claimed here."
        )
    elif len(measured_seeds) == len(seeds):
        note = "Every measured paired/augmented ratio is at most 1.10."
    else:
        note = (
            "Paired/augmented timing ratios are unavailable for some seeds; "
            "no compute-matched claim is made."
        )
    return {
        "paired_lambda": paired_lambda,
        "by_seed": by_seed,
        "measured_seeds": measured_seeds,
        "compute_matched_augmented_control_required": required,
        "note": note,
    }


def _diagnostic_calibration_rows(
    records: list[dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Select ordered original/option-permutation calibration rows only."""
    pairs = select_calibration_pairs(records)
    ordered = [record for pair in pairs for record in pair.ordered_records()]
    return pairs, ordered


def _validate_diagnostic_offsets(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    config: dict[str, Any],
    offsets_path: Path,
) -> None:
    if metadata.get("format") != DIAGNOSTIC_FORMAT:
        raise ValueError("diagnostic metadata format is unsupported")
    if metadata.get("state") != "completed":
        raise RuntimeError(
            f"diagnostic is not completed: state={metadata.get('state')!r}"
        )
    _check_config(config, metadata.get("config"), "diagnostic")
    _check_model(config, metadata.get("model"), "diagnostic")
    if metadata.get("objective") != RTN_BIAS_OBJECTIVE:
        raise ValueError("diagnostic objective mismatch")
    if metadata.get("fit_seed") is not None:
        raise ValueError("diagnostic must be one deterministic rule without a seed")
    if metadata.get("calibration_split") != "calibration":
        raise ValueError("diagnostic offsets must be fitted on calibration only")
    if list(metadata.get("bits", [])) != list(DIAGNOSTIC_BITS):
        raise ValueError("diagnostic must use only the fixed W4 bit width")
    if int(metadata.get("record_count", -1)) != DIAGNOSTIC_RECORD_COUNT:
        raise ValueError(
            "diagnostic record count must be "
            f"{DIAGNOSTIC_RECORD_COUNT}, got {metadata.get('record_count')!r}"
        )
    if str(metadata.get("offsets_path")) != str(offsets_path):
        raise ValueError("diagnostic offsets path mismatch")

    apply_offsets([], payload)

    if payload.get("condition") != RTN_BIAS_CONDITION:
        raise ValueError("diagnostic offsets condition mismatch")
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        raise ValueError("diagnostic offsets lack fit metadata")
    if fit.get("split") != "calibration":
        raise ValueError("diagnostic offsets were not fitted on calibration")
    if list(fit.get("transforms", [])) != ["original", "option_permutation"]:
        raise ValueError("diagnostic offsets used unexpected transforms")
    if int(fit.get("record_count", -1)) != DIAGNOSTIC_RECORD_COUNT:
        raise ValueError("diagnostic offsets were fitted on a different record count")
    if metadata.get("offsets") != payload.get("offsets"):
        raise ValueError("diagnostic metadata offsets do not match the fit")


def _load_diagnostic_offsets(
    config: dict[str, Any],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = _diagnostic_metadata_path(root)
    offsets_path = _diagnostic_offsets_path(root)
    if not metadata_path.is_file() or not offsets_path.is_file():
        raise FileNotFoundError(
            "completed diagnostic offsets are missing; run the diagnostic phase first"
        )
    metadata = _read_json(metadata_path)
    payload = _read_json(offsets_path)
    _validate_diagnostic_offsets(payload, metadata, config, offsets_path)
    return payload, metadata


def _load_completed_diagnostic(
    config: dict[str, Any],
    root: Path,
    calibration_records_path: Path,
    expected_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostic_dir = _diagnostic_dir(root)
    evaluation_dir = _diagnostic_evaluation_dir(root)
    _require_files(
        diagnostic_dir,
        [
            DIAGNOSTIC_RECORDS_FILENAME,
            DIAGNOSTIC_METADATA_FILENAME,
            DIAGNOSTIC_OFFSETS_FILENAME,
        ],
        "diagnostic",
    )
    _require_files(
        evaluation_dir,
        ["run.json", "predictions_fp.jsonl", "predictions_w4.jsonl"],
        "diagnostic evaluation",
    )

    payload, metadata = _load_diagnostic_offsets(config, root)
    evaluation_manifest = _read_json(evaluation_dir / "run.json")
    _check_model(config, evaluation_manifest.get("model"), "diagnostic evaluation")
    if evaluation_manifest.get("state") != "completed":
        raise RuntimeError(
            "diagnostic evaluation is not completed: "
            f"state={evaluation_manifest.get('state')!r}"
        )
    bits = [
        int(value)
        for value in evaluation_manifest.get("quantization", {}).get("bits", [])
    ]
    if bits != list(DIAGNOSTIC_BITS):
        raise ValueError("diagnostic evaluation used different quantization bits")
    if sorted(evaluation_manifest.get("completed_conditions", [])) != ["fp", "w4"]:
        raise ValueError("diagnostic evaluation did not complete only FP and W4")
    if str(metadata.get("source_calibration_records")) != str(
        calibration_records_path
    ):
        raise ValueError("diagnostic was built from different calibration records")

    saved_rows = load_records(_diagnostic_records_path(root))
    if saved_rows != expected_rows:
        raise ValueError(
            "diagnostic inputs do not match the current calibration originals/"
            "option permutations"
        )
    return payload, metadata


def run_diagnostic(
    config: dict[str, Any],
    root: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    calibration_dir = root / "data" / "calibration"
    _require_files(
        calibration_dir,
        ["records.jsonl", "manifest.json", "study_data.json"],
        "calibration dataset",
    )
    calibration_metadata = _read_json(calibration_dir / "study_data.json")
    if calibration_metadata.get("format") != METADATA_FORMAT:
        raise ValueError(
            f"calibration dataset metadata format is unsupported: {calibration_dir}"
        )
    _check_config(config, calibration_metadata.get("config"), "calibration dataset")
    calibration_records_path = calibration_dir / "records.jsonl"

    all_records = load_records(calibration_records_path)
    pairs, ordered_rows = _diagnostic_calibration_rows(all_records)
    if (
        len(ordered_rows) != DIAGNOSTIC_RECORD_COUNT
        or len(pairs) * 2 != DIAGNOSTIC_RECORD_COUNT
    ):
        raise ValueError(
            "diagnostic requires exactly "
            f"{DIAGNOSTIC_RECORD_COUNT} calibration original/option rows; "
            f"found {len(ordered_rows)} rows from {len(pairs)} bases"
        )

    diagnostic_dir = _diagnostic_dir(root)
    records_path = _diagnostic_records_path(root)
    evaluation_dir = _diagnostic_evaluation_dir(root)
    offsets_path = _diagnostic_offsets_path(root)
    metadata_path = _diagnostic_metadata_path(root)

    if diagnostic_dir.exists():
        _log("diagnostic", f"reuse completed diagnostic {diagnostic_dir}")
        payload, metadata = _load_completed_diagnostic(
            config, root, calibration_records_path, ordered_rows
        )
        return {
            "inputs": str(records_path),
            "evaluation": str(evaluation_dir),
            "offsets": str(offsets_path),
            "metadata": str(metadata_path),
            "record_count": int(metadata["record_count"]),
            "calibration_bases": int(metadata["calibration_bases"]),
        }

    _log(
        "diagnostic",
        f"save {len(ordered_rows)} calibration original/option rows -> {records_path}",
    )
    diagnostic_dir.mkdir(parents=True)
    _write_jsonl_no_overwrite(records_path, ordered_rows)

    diagnostic_config = copy.deepcopy(config)
    diagnostic_config["quantization"]["bits"] = list(DIAGNOSTIC_BITS)
    _log(
        "diagnostic",
        f"evaluate FP/W4 calibration rows -> {evaluation_dir}",
    )
    evaluation_manifest = run_evaluation(
        config=diagnostic_config,
        records_path=records_path,
        output_dir=evaluation_dir,
        generation_subset=0,
        overwrite=False,
    )
    if evaluation_manifest.get("state") != "completed":
        raise RuntimeError(
            "diagnostic evaluation did not complete: "
            f"state={evaluation_manifest.get('state')!r}"
        )

    fp_path = evaluation_dir / "predictions_fp.jsonl"
    w4_path = evaluation_dir / "predictions_w4.jsonl"
    fp_rows = load_predictions(fp_path)
    w4_rows = load_predictions(w4_path)
    if (
        len(fp_rows) != DIAGNOSTIC_RECORD_COUNT
        or len(w4_rows) != DIAGNOSTIC_RECORD_COUNT
    ):
        raise ValueError(
            "diagnostic evaluation returned an unexpected prediction count"
        )

    _log("diagnostic", "fit fixed four-label offsets from calibration only")
    payload = fit_offsets(
        fp_rows,
        w4_rows,
        config=config,
        fp_path=fp_path,
        w4_path=w4_path,
    )
    fit = payload.get("fit", {})
    if int(fit.get("record_count", -1)) != DIAGNOSTIC_RECORD_COUNT:
        raise ValueError("diagnostic offset fit used an unexpected record count")
    _write_json_no_overwrite(offsets_path, payload)

    metadata = {
        "format": DIAGNOSTIC_FORMAT,
        "kind": "diagnostic",
        "state": "completed",
        "phase": "diagnostic",
        "objective": RTN_BIAS_OBJECTIVE,
        "condition": RTN_BIAS_CONDITION,
        "fit_seed": None,
        "secondary": True,
        "config": config,
        "model": config["model"],
        "diagnostic_config": diagnostic_config,
        "bits": list(DIAGNOSTIC_BITS),
        "calibration_split": "calibration",
        "calibration_bases": len(pairs),
        "record_count": len(ordered_rows),
        "records_path": str(records_path),
        "source_calibration_records": str(calibration_records_path),
        "evaluation_path": str(evaluation_dir),
        "offsets_path": str(offsets_path),
        "offsets": payload["offsets"],
        "targets": {
            "uses_oracle_labels": False,
            "uses_development_or_final_outcomes": False,
        },
        "source": provenance,
        "created_at": _now(),
    }
    _write_json_no_overwrite(metadata_path, metadata)
    _log("diagnostic", f"wrote diagnostic offsets {offsets_path}")
    return {
        "inputs": str(records_path),
        "evaluation": str(evaluation_dir),
        "offsets": str(offsets_path),
        "metadata": str(metadata_path),
        "record_count": len(ordered_rows),
        "calibration_bases": len(pairs),
    }


def run_develop(
    config: dict[str, Any],
    root: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    seeds = _fit_seeds(config)
    lambdas = _lambda_candidates(config)

    calibration_records = _ensure_dataset(
        config, root / "data" / "calibration", ["calibration"], provenance, "develop"
    )
    development_records = _ensure_dataset(
        config, root / "data" / "development", ["development"], provenance, "develop"
    )
    teacher_cache = _ensure_teacher_cache(
        config, calibration_records, root / "teacher-cache", provenance, "develop"
    )

    controls: dict[str, dict[str, Any]] = {}
    for objective in CONTROL_OBJECTIVES:
        fit_dir_path = _fit_dir(root, objective, None, DEVELOPMENT_SEED)
        _ensure_fit(
            config,
            calibration_records,
            teacher_cache,
            fit_dir_path,
            objective,
            None,
            DEVELOPMENT_SEED,
            "develop",
        )
        evaluation_dir = _development_dir(root, objective, None, DEVELOPMENT_SEED)
        condition = _development_condition(objective, None, DEVELOPMENT_SEED)
        _ensure_calibrated_evaluation(
            config,
            development_records,
            fit_dir_path,
            evaluation_dir,
            objective,
            None,
            DEVELOPMENT_SEED,
            condition,
            provenance,
            "develop",
        )
        controls[objective] = {
            "fit_seed": DEVELOPMENT_SEED,
            "condition": condition,
            "fit_path": str(fit_dir_path),
            "evaluation_path": str(evaluation_dir),
            "metrics": _development_metrics(config, evaluation_dir, condition),
        }

    augmented_accuracy = controls["augmented_kd"]["metrics"]["overall_accuracy"]
    if augmented_accuracy is None:
        raise ValueError("augmented KD development accuracy is unavailable")

    objectives: dict[str, dict[str, Any]] = {}
    for objective in REGULARIZED_OBJECTIVES:
        candidates: list[dict[str, Any]] = []
        for lambda_value in lambdas:
            fit_dir_path = _fit_dir(root, objective, lambda_value, DEVELOPMENT_SEED)
            _ensure_fit(
                config,
                calibration_records,
                teacher_cache,
                fit_dir_path,
                objective,
                lambda_value,
                DEVELOPMENT_SEED,
                "develop",
            )
            evaluation_dir = _development_dir(
                root, objective, lambda_value, DEVELOPMENT_SEED
            )
            condition = _development_condition(
                objective, lambda_value, DEVELOPMENT_SEED
            )
            _ensure_calibrated_evaluation(
                config,
                development_records,
                fit_dir_path,
                evaluation_dir,
                objective,
                lambda_value,
                DEVELOPMENT_SEED,
                condition,
                provenance,
                "develop",
            )
            metrics = _development_metrics(config, evaluation_dir, condition)
            candidates.append(
                {
                    "lambda": float(lambda_value),
                    "condition": condition,
                    "fit_path": str(fit_dir_path),
                    "evaluation_path": str(evaluation_dir),
                    **metrics,
                }
            )

        selection_result = select_lambda(candidates, augmented_accuracy)
        selected_lambda = float(selection_result["selected_lambda"])
        selected = next(
            candidate
            for candidate in candidates
            if candidate["lambda"] == selected_lambda
        )
        objectives[objective] = {
            "selected_lambda": selected_lambda,
            "selected": selected,
            "selection": selection_result,
            "candidates": candidates,
            "condition": selected["condition"],
            "fit_path": selected["fit_path"],
            "evaluation_path": selected["evaluation_path"],
        }

    selection = {
        "format": SELECTION_FORMAT,
        "state": "frozen",
        "model": config["model"],
        "config": config,
        "source": provenance,
        "source_revision": provenance["git_revision"],
        "source_dirty_paths": provenance["dirty_paths"],
        "created_at": _now(),
        "development_seed": DEVELOPMENT_SEED,
        "fit_seeds": seeds,
        "lambda_candidates": lambdas,
        "controls": controls,
        "objectives": objectives,
        "paths": {
            "root": str(root),
            "calibration_records": str(calibration_records),
            "development_records": str(development_records),
            "teacher_cache": str(teacher_cache),
            "fits": str(root / "fits"),
            "development": str(root / "development"),
        },
    }
    selection_path = root / "selection.json"
    write_json(selection_path, selection)
    _log("develop", f"wrote frozen selection {selection_path}")

    timing = _optimization_time_check(root, selection, [DEVELOPMENT_SEED])
    if timing["compute_matched_augmented_control_required"]:
        _log(
            "develop",
            "WARNING paired/augmented optimization-time ratio exceeds 1.10; "
            "a compute-matched augmented control is required",
        )
    return {
        "selection": str(selection_path),
        "calibration_records": str(calibration_records),
        "development_records": str(development_records),
        "teacher_cache": str(teacher_cache),
        "optimization_time": timing,
    }


def run_replicate(
    config: dict[str, Any],
    root: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    seeds = _fit_seeds(config)
    selection = _load_selection(root, config)
    calibration_records = root / "data" / "calibration" / "records.jsonl"
    teacher_cache = root / "teacher-cache" / CACHE_FILENAME
    _require_files(
        root / "data" / "calibration",
        ["records.jsonl", "manifest.json", "study_data.json"],
        "calibration dataset",
    )
    _require_files(
        root / "teacher-cache",
        [CACHE_FILENAME, CACHE_MANIFEST, "study_cache.json"],
        "teacher cache",
    )

    for entry in _selected_entries(selection):
        _load_fit(
            config,
            _fit_dir(root, entry["objective"], entry["lambda"], DEVELOPMENT_SEED),
            entry["objective"],
            entry["lambda"],
            DEVELOPMENT_SEED,
        )

    completed: list[str] = []
    replicate_seeds = [seed for seed in seeds if seed != DEVELOPMENT_SEED]
    for entry in _selected_entries(selection):
        for seed in replicate_seeds:
            fit_dir_path = _fit_dir(root, entry["objective"], entry["lambda"], seed)
            _ensure_fit(
                config,
                calibration_records,
                teacher_cache,
                fit_dir_path,
                entry["objective"],
                entry["lambda"],
                seed,
                "replicate",
            )
            completed.append(str(fit_dir_path))

    timing = _optimization_time_check(root, selection, seeds)
    if timing["compute_matched_augmented_control_required"]:
        _log(
            "replicate",
            "WARNING paired/augmented optimization-time ratio exceeds 1.10; "
            "a compute-matched augmented control is required",
        )
    return {
        "selection": str(root / "selection.json"),
        "fits": completed,
        "optimization_time": timing,
    }


def run_final(
    config: dict[str, Any],
    root: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    seeds = _fit_seeds(config)
    selection = _load_selection(root, config)
    selected_fits = _require_selected_fits(config, root, selection, seeds)
    _log("final", f"verified {len(selected_fits)} completed selected fits")
    offsets_payload, _ = _load_diagnostic_offsets(config, root)
    _log("final", "verified completed diagnostic offsets before final inference")

    records_path = _ensure_dataset(
        config,
        root / "data" / "final_structural",
        ["final", "structural_holdout"],
        provenance,
        "final",
    )
    baseline_dir = root / "results" / "final" / "baseline"
    _ensure_baseline_evaluation(
        config, records_path, baseline_dir, provenance, "final"
    )
    rtn_bias_dir = _rtn_bias_dir(root)
    _ensure_rtn_bias_control(
        config,
        records_path,
        baseline_dir,
        rtn_bias_dir,
        _diagnostic_offsets_path(root),
        offsets_payload,
        _diagnostic_metadata_path(root),
        provenance,
        "final",
    )

    evaluations: list[str] = []
    for entry in selected_fits:
        fit_dir_path = root / "fits" / _version(
            entry["objective"], entry["lambda"]
        ) / f"seed-{entry['seed']}"
        evaluation_dir = _final_dir(
            root, entry["objective"], entry["lambda"], entry["seed"]
        )
        condition = _final_condition(entry["objective"], entry["seed"])
        _ensure_calibrated_evaluation(
            config,
            records_path,
            fit_dir_path,
            evaluation_dir,
            entry["objective"],
            entry["lambda"],
            entry["seed"],
            condition,
            provenance,
            "final",
        )
        evaluations.append(str(evaluation_dir))

    constant_control_dir = _constant_display_a_dir(root)
    _ensure_constant_display_a_control(
        config,
        records_path,
        constant_control_dir,
        provenance,
        "final",
    )

    return {
        "selection": str(root / "selection.json"),
        "records": str(records_path),
        "baseline": str(baseline_dir),
        "evaluations": evaluations,
        "rtn_w4_logit_bias": str(rtn_bias_dir),
        "constant_display_a": str(constant_control_dir),
    }


def _held_out_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transforms = {ORIGINAL_TRANSFORM, *HELD_OUT_TRANSFORMS}
    return [row for row in rows if str(row.get("transform")) in transforms]


def _composition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transforms = {ORIGINAL_TRANSFORM, *COMPOSED_TRANSFORMS}
    return [row for row in rows if str(row.get("transform")) in transforms]


def _filter_rows(
    rows: list[dict[str, Any]],
    split: str | None = None,
    task_family: str | None = None,
    condition: str | None = None,
) -> list[dict[str, Any]]:
    selected = rows
    if split is not None:
        selected = [row for row in selected if row["split"] == split]
    if task_family is not None:
        selected = [
            row for row in selected if row["task_family"] == task_family
        ]
    if condition is not None:
        selected = [row for row in selected if row["condition"] == condition]
    return selected


def _without_bases(
    summaries: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        condition: {
            key: value for key, value in summary.items() if key != "bases"
        }
        for condition, summary in summaries.items()
    }


def _group_summaries(
    rows: list[dict[str, Any]],
    conditions: list[str],
    splits: list[str],
    families: list[str],
    seed_by_condition: dict[str, int],
) -> dict[str, Any]:
    by_split = {
        split: _without_bases(
            summarize_conditions(
                _filter_rows(rows, split=split),
                [
                    condition
                    for condition in conditions
                    if _filter_rows(rows, split=split, condition=condition)
                ],
            )
        )
        for split in splits
    }
    by_task = {
        family: _without_bases(
            summarize_conditions(
                _filter_rows(rows, task_family=family),
                [
                    condition
                    for condition in conditions
                    if _filter_rows(rows, task_family=family, condition=condition)
                ],
            )
        )
        for family in families
    }
    by_seed = {
        str(seed): _without_bases(
            summarize_conditions(
                rows,
                [
                    condition
                    for condition in conditions
                    if seed_by_condition.get(condition) == seed
                ],
            )
        )
        for seed in sorted(set(seed_by_condition.values()))
    }
    by_split_task = {
        split: {
            family: _without_bases(
                summarize_conditions(
                    _filter_rows(rows, split=split, task_family=family),
                    [
                        condition
                        for condition in conditions
                        if _filter_rows(
                            rows,
                            split=split,
                            task_family=family,
                            condition=condition,
                        )
                    ],
                )
            )
            for family in families
        }
        for split in splits
    }
    return {
        "by_split": by_split,
        "by_task_family": by_task,
        "by_seed": by_seed,
        "by_split_task_family": by_split_task,
    }


def _comparison_rows(
    rows: list[dict[str, Any]],
    evidence: str,
) -> list[dict[str, Any]]:
    if evidence == "held_out_singles":
        return _held_out_rows(rows)
    if evidence == "compositions":
        return _composition_rows(rows)
    raise ValueError(f"unknown comparison evidence {evidence!r}")


def _primary_comparisons(
    rows: list[dict[str, Any]],
    seeds: list[int],
    splits: list[str],
    families: list[str],
    n_boot: int,
    bootstrap_seed: int,
    evidence: str = "held_out_singles",
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    scopes: list[tuple[str | None, str | None]] = [(None, None)]
    scopes.extend((split, None) for split in splits)
    scopes.extend((None, family) for family in families)
    scopes.extend(
        (split, family) for split in splits for family in families
    )
    compared_rows = _comparison_rows(rows, evidence)
    for seed in seeds:
        condition = _final_condition("paired", seed)
        baseline = _final_condition("augmented_kd", seed)
        for split, family in scopes:
            selected = _filter_rows(
                compared_rows, split=split, task_family=family
            )
            available = {row["condition"] for row in selected}
            if condition not in available or baseline not in available:
                continue
            if evidence == "compositions" and not any(
                str(row.get("transform")) in COMPOSED_TRANSFORMS
                for row in selected
            ):
                continue
            comparison = compare_conditions(
                selected,
                condition,
                baseline,
                reference="fp",
                n_boot=n_boot,
                seed=bootstrap_seed,
            )
            comparisons.append(
                {
                    "fit_seed": seed,
                    "split": split or "all",
                    "task_family": family or "all",
                    "condition": condition,
                    "baseline": baseline,
                    "evidence": evidence,
                    "comparison": comparison,
                }
            )
    return comparisons


def _require_final_rows(
    config: dict[str, Any],
    root: Path,
    selection: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seeds = _fit_seeds(config)
    selected_fits = _require_selected_fits(config, root, selection, seeds)
    records_path = root / "data" / "final_structural" / "records.jsonl"
    _require_files(
        root / "data" / "final_structural",
        ["records.jsonl", "manifest.json", "study_data.json"],
        "final dataset",
    )

    baseline_dir = root / "results" / "final" / "baseline"
    _load_baseline_evaluation(config, baseline_dir, records_path)
    rows = load_predictions(baseline_dir)

    condition_index: dict[str, dict[str, Any]] = {}
    for baseline_condition in sorted({row["condition"] for row in rows}):
        condition_index[baseline_condition] = {
            "objective": "baseline",
            "lambda": None,
            "fit_seed": None,
            "path": str(baseline_dir),
        }

    for entry in selected_fits:
        evaluation_dir = _final_dir(
            root, entry["objective"], entry["lambda"], entry["seed"]
        )
        condition = _final_condition(entry["objective"], entry["seed"])
        _load_calibrated_evaluation(
            config,
            evaluation_dir,
            records_path,
            root / "fits" / _version(entry["objective"], entry["lambda"])
            / f"seed-{entry['seed']}",
            entry["objective"],
            entry["lambda"],
            entry["seed"],
            condition,
        )
        if condition in condition_index:
            raise ValueError(f"duplicate final condition: {condition}")
        condition_index[condition] = {
            "objective": entry["objective"],
            "lambda": entry["lambda"],
            "fit_seed": entry["seed"],
            "path": str(evaluation_dir),
        }
        rows.extend(load_predictions(evaluation_dir))

    offsets_payload, _ = _load_diagnostic_offsets(config, root)
    rtn_bias_dir = _rtn_bias_dir(root)
    _load_rtn_bias_control(
        config,
        rtn_bias_dir,
        records_path,
        _diagnostic_offsets_path(root),
        offsets_payload,
    )
    if RTN_BIAS_CONDITION in condition_index:
        raise ValueError(f"duplicate final condition: {RTN_BIAS_CONDITION}")
    condition_index[RTN_BIAS_CONDITION] = {
        "objective": RTN_BIAS_OBJECTIVE,
        "lambda": None,
        "fit_seed": None,
        "path": str(rtn_bias_dir),
        "secondary_postprocessing": True,
        "manifest": str(rtn_bias_dir / POSTPROCESSING_MANIFEST),
    }
    rows.extend(load_predictions(rtn_bias_dir))

    constant_control_dir = _constant_display_a_dir(root)
    _load_constant_display_a_control(config, constant_control_dir, records_path)
    if CONSTANT_DISPLAY_A_CONDITION in condition_index:
        raise ValueError(
            f"duplicate final condition: {CONSTANT_DISPLAY_A_CONDITION}"
        )
    condition_index[CONSTANT_DISPLAY_A_CONDITION] = {
        "objective": "constant_display_A",
        "lambda": None,
        "fit_seed": None,
        "path": str(constant_control_dir),
        "untrained_rule": True,
        "manifest": str(constant_control_dir / UNTRAINED_RULE_MANIFEST),
    }
    rows.extend(load_predictions(constant_control_dir))

    return rows, condition_index


def _per_transform_evidence(
    rows: list[dict[str, Any]],
    conditions: list[str],
) -> dict[str, dict[str, Any]]:
    """Report accuracy for every condition and transform without aggregating."""
    evidence: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        condition_rows = [
            row for row in rows if row["condition"] == condition
        ]
        transforms = sorted({row["transform"] for row in condition_rows})
        evidence[condition] = {}
        for transform in transforms:
            selected = [
                row for row in condition_rows if row["transform"] == transform
            ]
            n_correct = sum(int(recompute_correct(row)) for row in selected)
            evidence[condition][transform] = {
                "n_records": len(selected),
                "n_correct": n_correct,
                "accuracy": n_correct / len(selected),
                "splits": sorted({row["split"] for row in selected}),
            }
    return evidence


def run_report(
    config: dict[str, Any],
    root: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    seeds = _fit_seeds(config)
    selection = _load_selection(root, config)
    rows, condition_index = _require_final_rows(config, root, selection)

    report_dir = root / "report"
    report_manifest = report_dir / "report_manifest.json"
    expected = [
        "final_predictions.jsonl",
        "summaries.json",
        "comparisons.json",
        "report_manifest.json",
    ]
    if report_dir.exists():
        _require_files(report_dir, expected, "report")
        metadata = _read_json(report_manifest)
        _check_config(config, metadata.get("config"), "report")
        _check_model(config, metadata.get("model"), "report")
        if metadata.get("conditions") != sorted(condition_index):
            raise ValueError(f"report conditions do not match final results: {report_dir}")
        _log("report", f"reuse report {report_dir}")
        return {
            "report": str(report_dir),
            "summaries": str(report_dir / "summaries.json"),
            "comparisons": str(report_dir / "comparisons.json"),
            "predictions": str(report_dir / "final_predictions.jsonl"),
        }

    conditions = sorted(condition_index)
    summaries = summarize_conditions(rows, conditions)
    splits = sorted({row["split"] for row in rows})
    families = sorted({row["task_family"] for row in rows})
    seed_by_condition = {
        condition: int(entry["fit_seed"])
        for condition, entry in condition_index.items()
        if entry["fit_seed"] is not None
    }
    grouped = _group_summaries(
        rows, conditions, splits, families, seed_by_condition
    )
    grouped["by_model"] = {
        config["model"]["id"]: _without_bases(summaries)
    }

    analysis_config = config.get("analysis", {})
    n_boot = int(analysis_config.get("bootstrap_samples", 1000))
    bootstrap_seed = int(analysis_config.get("seed", 4108))
    comparisons = _primary_comparisons(
        rows, seeds, splits, families, n_boot, bootstrap_seed
    )
    secondary_compositions = _primary_comparisons(
        rows,
        seeds,
        splits,
        families,
        n_boot,
        bootstrap_seed,
        evidence="compositions",
    )
    per_transform = _per_transform_evidence(rows, conditions)
    untrained_rule_conditions = [
        condition
        for condition, entry in condition_index.items()
        if entry.get("untrained_rule")
    ]
    secondary_postprocessing_conditions = [
        condition
        for condition, entry in condition_index.items()
        if entry.get("secondary_postprocessing")
    ]
    timing = _optimization_time_check(root, selection, seeds)

    summaries_payload = {
        "format": REPORT_FORMAT,
        "state": "completed",
        "model": config["model"],
        "config": config,
        "source": provenance,
        "created_at": _now(),
        "conditions": conditions,
        "condition_index": condition_index,
        "summaries": summaries,
        "grouped_summaries": grouped,
        "per_transform": per_transform,
        "descriptive_control_conditions": [
            condition
            for condition, entry in condition_index.items()
            if entry["objective"]
            in {"baseline", "original_kd", "random_pair", "kernel"}
            or entry["objective"] == RTN_BIAS_OBJECTIVE
            or entry.get("untrained_rule")
        ],
        "untrained_rule_conditions": untrained_rule_conditions,
        "secondary_postprocessing_conditions": secondary_postprocessing_conditions,
        "optimization_time": timing,
        "paths": {
            "prediction_rows": str(report_dir / "final_predictions.jsonl"),
            "comparisons": str(report_dir / "comparisons.json"),
        },
    }
    comparisons_payload = {
        "format": REPORT_FORMAT,
        "state": "completed",
        "model": config["model"],
        "config": config,
        "source": provenance,
        "created_at": _now(),
        "primary": "paired_vs_augmented_kd_same_seed",
        "evidence": "held_out_singles",
        "reference_condition": "fp",
        "bootstrap": {
            "unit": "base_id",
            "samples": n_boot,
            "seed": bootstrap_seed,
        },
        "endpoint_descriptions": {
            "primary": {
                "key": "comparisons",
                "evidence": "held_out_singles",
                "transforms": [ORIGINAL_TRANSFORM, *HELD_OUT_TRANSFORMS],
                "description": (
                    "Original plus entity-rename/order single transformations. "
                    "Entity rename and order were excluded from gradients but "
                    "used in lambda selection; these contrasts are not free of "
                    "development selection exposure."
                ),
            },
            "secondary_compositions": {
                "key": "secondary_compositions",
                "evidence": "compositions",
                "transforms": [ORIGINAL_TRANSFORM, *COMPOSED_TRANSFORMS],
                "description": (
                    "Original plus option+rename and option+rename+order "
                    "compositions. Compositions are withheld from both "
                    "calibration and development, so these contrasts add "
                    "validation-free composition evidence; they remain "
                    "secondary to the primary held-out comparison."
                ),
            },
        },
        "comparisons": comparisons,
        "secondary_compositions": secondary_compositions,
        "optimization_time": timing,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(report_dir / "final_predictions.jsonl", rows)
    write_json(report_dir / "summaries.json", summaries_payload)
    write_json(report_dir / "comparisons.json", comparisons_payload)
    write_json(
        report_manifest,
        {
            "format": REPORT_FORMAT,
            "state": "completed",
            "model": config["model"],
            "config": config,
            "source": provenance,
            "created_at": _now(),
            "conditions": conditions,
        },
    )
    _log("report", f"wrote report {report_dir}")
    return {
        "report": str(report_dir),
        "summaries": str(report_dir / "summaries.json"),
        "comparisons": str(report_dir / "comparisons.json"),
        "predictions": str(report_dir / "final_predictions.jsonl"),
    }


def run_phase(
    config: dict[str, Any],
    root: Path,
    phase: str,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    root.mkdir(parents=True, exist_ok=True)
    provenance = _provenance()
    _log(
        phase,
        f"starting root={root} git={provenance['git_revision']} "
        f"dirty_paths={len(provenance['dirty_paths'])}",
    )
    try:
        if phase == "develop":
            result = run_develop(config, root, provenance)
        elif phase == "replicate":
            result = run_replicate(config, root, provenance)
        elif phase == "diagnostic":
            result = run_diagnostic(config, root, provenance)
        elif phase == "final":
            result = run_final(config, root, provenance)
        else:
            result = run_report(config, root, provenance)
    except Exception as error:
        _append_journal(
            root,
            {
                "phase": phase,
                "status": "failed",
                "timestamp": _now(),
                "source": provenance,
                "source_revision": provenance["git_revision"],
                "source_dirty_paths": provenance["dirty_paths"],
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise

    _append_journal(
        root,
        {
            "phase": phase,
            "status": "completed",
            "timestamp": _now(),
            "source": provenance,
            "source_revision": provenance["git_revision"],
            "source_dirty_paths": provenance["dirty_paths"],
            "outputs": result,
        },
    )
    _log(phase, f"completed; outputs: {json.dumps(result, sort_keys=True)}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eq_quant.study",
        description="Run one explicit phase of the frozen calibration study.",
    )
    parser.add_argument("--config", required=True, help="study JSON config")
    parser.add_argument("--root", required=True, help="study output root")
    parser.add_argument("--phase", required=True, choices=list(PHASES))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    run_phase(config, Path(args.root), args.phase)
    return 0


if __name__ == "__main__":
    sys.exit(main())
