"""Score arbitrary records with exported calibrated RTN scales."""

import time
from pathlib import Path
from typing import Any

import torch

from eq_quant.evaluate.runner import (
    load_model,
    load_records,
    set_determinism,
    verify_records,
)
from eq_quant.evaluate.scoring import (
    build_chat_prompt,
    score_records,
    verify_option_token_ids,
)

from .basis import build_rtn_basis, load_scales, replace_linear_modules
from .io_helpers import (
    hardware_metadata,
    peak_memory_bytes,
    prepare_output_files,
    reset_peak_memory,
    software_metadata,
    write_json,
    write_jsonl,
)

MANIFEST_FILENAME = "run.json"
PREDICTIONS_FILENAME = "predictions.jsonl"


def _load_scale_payload(scales_path: str | Path) -> dict[str, Any]:
    path = Path(scales_path)
    if path.is_dir():
        path = path / "scales.pt"
    if not path.is_file():
        raise FileNotFoundError(f"calibration scales not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "eq_quant_calibration_scales_v1":
        raise ValueError(f"unrecognised calibration scale format in {path}")
    return payload


def run_calibrated_evaluation(
    config: dict[str, Any],
    records_path: str | Path,
    scales_path: str | Path,
    output_dir: str | Path,
    condition: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not condition or not condition.strip():
        raise ValueError("an explicit --condition name is required")

    payload = _load_scale_payload(scales_path)
    if payload.get("scale_parameterization") != "rtn_group_scale_times_exp_delta":
        raise ValueError(
            "unsupported calibration scale parameterization: "
            f"{payload.get('scale_parameterization')!r}"
        )
    scale_model = payload["model"]
    if scale_model["id"] != config["model"]["id"]:
        raise ValueError("calibration scales were fit on a different model id")
    if scale_model["revision"] != config["model"]["revision"]:
        raise ValueError("calibration scales were fit on a different model revision")

    bits = int(payload["bits"])
    group_size = int(payload["group_size"])
    if group_size != int(config["quantization"]["group_size"]):
        raise ValueError("scale group size does not match the config")
    if bits not in [int(value) for value in config["quantization"]["bits"]]:
        raise ValueError("scale bit width is not enabled in the config")

    records = load_records(records_path)
    if not records:
        raise ValueError("no records to evaluate")
    verify_records(records)

    output_path = prepare_output_files(
        output_dir, [MANIFEST_FILENAME, PREDICTIONS_FILENAME], overwrite
    )
    manifest: dict[str, Any] = {
        "state": "running",
        "condition": condition,
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "dtype": config["model"]["dtype"],
            "resolved_commit_hash": None,
        },
        "calibration": {
            "source": str(scales_path),
            "objective": payload.get("objective"),
            "lambda": payload.get("lambda"),
            "fit_seed": payload.get("fit_seed"),
            "scale_parameterization": payload.get("scale_parameterization"),
            "bits": bits,
            "group_size": group_size,
            "codes": "pinned original symmetric signed RTN integer codes",
        },
        "records": {"path": str(records_path), "count": len(records)},
        "software": software_metadata(),
        "hardware": None,
        "elapsed_seconds": None,
        "peak_memory_bytes": None,
        "error": None,
    }
    write_json(output_path / MANIFEST_FILENAME, manifest)

    model = None
    try:
        started = time.perf_counter()
        set_determinism(
            seed=int(config.get("analysis", {}).get("seed", 4108)),
            cpu_threads=int(config["scoring"].get("cpu_threads", 6)),
        )
        model, tokenizer, device = load_model(config)
        tokenizer.padding_side = "left"
        manifest["model"]["resolved_commit_hash"] = getattr(
            model.config, "_commit_hash", None
        )
        manifest["hardware"] = hardware_metadata(device)

        option_token_ids = verify_option_token_ids(
            tokenizer,
            labels=tuple(config["scoring"]["option_labels"]),
            sample_context=build_chat_prompt(tokenizer, records[0]["prompt"]),
        )

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        basis = build_rtn_basis(model, bits, group_size)
        modules = replace_linear_modules(model, basis)
        load_scales(modules, payload["scale_deltas"])
        model.eval()
        reset_peak_memory(device)

        predictions = score_records(
            model=model,
            tokenizer=tokenizer,
            records=records,
            condition=condition,
            option_token_ids=option_token_ids,
            batch_size=int(config["scoring"]["batch_size"]),
            max_seq_len=int(config["scoring"]["max_seq_len"]),
        )
        write_jsonl(output_path / PREDICTIONS_FILENAME, predictions)

        manifest["state"] = "completed"
        manifest["predictions"] = len(predictions)
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["peak_memory_bytes"] = peak_memory_bytes(device)
        write_json(output_path / MANIFEST_FILENAME, manifest)
    except Exception as error:
        manifest["state"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        if model is not None:
            manifest["peak_memory_bytes"] = peak_memory_bytes(
                next(model.parameters()).device
            )
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(output_path / MANIFEST_FILENAME, manifest)
        raise
    return manifest
