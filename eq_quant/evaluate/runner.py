"""End-to-end pilot evaluation runner (FP and RTN W8/W4/W3)."""

import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

from eq_quant.data.oracles import canonical_answer

from .quantize import (
    apply_rtn_to_model,
    fp_module_inventory,
    quantizable_linear_modules,
    restore_linear_weights,
    snapshot_linear_weights,
)
from .scoring import (
    build_chat_prompt,
    greedy_generation_subset,
    score_records,
    verify_option_token_ids,
)


def set_determinism(seed: int, cpu_threads: int = 6) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()


def load_model(config: dict[str, Any]) -> tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config["model"]["id"]
    revision = config["model"]["revision"]
    dtype_name = config["model"]["dtype"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records_path = Path(path)
    if records_path.is_dir():
        records_path = records_path / "records.jsonl"
    records: list[dict[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def verify_records(records: list[dict[str, Any]]) -> None:
    required = {
        "record_id",
        "base_id",
        "split",
        "task_family",
        "transform",
        "semantic",
        "prompt",
        "option_order",
        "correct_option_index",
    }
    for record in records:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"record {record.get('record_id')} is missing {missing}")
        correct_canonical = canonical_answer(
            record["task_family"],
            record["semantic"]["structure"],
            record["semantic"]["canonical_options"],
        )
        expected_display = list(record["option_order"]).index(correct_canonical)
        if expected_display != record["correct_option_index"]:
            raise ValueError(
                f"record {record['record_id']} failed oracle verification"
            )


def _software_metadata() -> dict[str, Any]:
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def _hardware_metadata(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        return {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(properties.total_memory),
        }
    return {"device": str(device), "device_name": platform.processor()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _append_jsonl(handle: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    handle.flush()


def _prepare_output_directory(
    output_path: Path,
    conditions: list[tuple[str, int | None]],
    overwrite: bool,
) -> None:
    protected = [
        output_path / "run.json",
        output_path / "predictions.jsonl",
        output_path / "generation.jsonl",
    ]
    for condition, _ in conditions:
        protected.append(output_path / f"predictions_{condition}.jsonl")
        protected.append(output_path / f"generation_{condition}.jsonl")
    if any(path.exists() for path in protected) and not overwrite:
        raise FileExistsError(
            f"{output_path} already contains evaluation output; pass overwrite=True"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in ("run.json", "predictions.jsonl", "generation.jsonl"):
            (output_path / name).unlink(missing_ok=True)
        for condition, _ in conditions:
            (output_path / f"predictions_{condition}.jsonl").unlink(missing_ok=True)
            (output_path / f"generation_{condition}.jsonl").unlink(missing_ok=True)


def run_evaluation(
    config: dict[str, Any],
    records_path: str | Path,
    output_dir: str | Path,
    generation_subset: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    records = load_records(records_path)
    if not records:
        raise ValueError("no records to evaluate")
    verify_records(records)

    group_size = int(config["quantization"]["group_size"])
    conditions: list[tuple[str, int | None]] = [("fp", None)]
    conditions.extend(
        (f"w{bits}", int(bits)) for bits in config["quantization"]["bits"]
    )

    output_path = Path(output_dir)
    _prepare_output_directory(output_path, conditions, overwrite)

    manifest: dict[str, Any] = {
        "state": "running",
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "resolved_commit_hash": None,
            "dtype": config["model"]["dtype"],
        },
        "software": _software_metadata(),
        "hardware": None,
        "quantization": {
            "method": "rtn_symmetric_signed_dequantized_weight_simulation",
            "group_size": group_size,
            "bits": [int(bits) for bits in config["quantization"]["bits"]],
            "excluded": ["lm_head", "embeddings"],
            "arithmetic": "dequantized weights used for BF16/FP16 arithmetic",
            "claim_limit": (
                "Numerical weight-only simulation; no packed-integer storage, "
                "deployment parity, or inference speed claim."
            ),
        },
        "scoring": {
            "option_labels": list(config["scoring"]["option_labels"]),
            "option_token_ids": None,
            "batch_size": int(config["scoring"]["batch_size"]),
            "max_seq_len": int(config["scoring"]["max_seq_len"]),
            "chat_template": "official tokenizer.apply_chat_template",
            "add_generation_prompt": True,
            "padding_side": "left",
        },
        "records": len(records),
        "conditions": [],
        "completed_conditions": [],
        "error": None,
    }
    _write_json(output_path / "run.json", manifest)

    model = None
    modules = None
    snapshot = None
    predictions_handle = None
    generation_handle = None
    try:
        set_determinism(
            seed=int(config["analysis"]["seed"]),
            cpu_threads=int(config["scoring"].get("cpu_threads", 6)),
        )
        model, tokenizer, device = load_model(config)
        tokenizer.padding_side = "left"

        sample_prompt = build_chat_prompt(tokenizer, records[0]["prompt"])
        option_token_ids = verify_option_token_ids(
            tokenizer,
            labels=tuple(config["scoring"]["option_labels"]),
            sample_context=sample_prompt,
        )
        manifest["model"]["resolved_commit_hash"] = getattr(
            model.config, "_commit_hash", None
        )
        manifest["hardware"] = _hardware_metadata(device)
        manifest["scoring"]["option_token_ids"] = list(option_token_ids)
        _write_json(output_path / "run.json", manifest)

        modules = quantizable_linear_modules(model)
        snapshot = snapshot_linear_weights(modules)
        predictions_handle = (output_path / "predictions.jsonl").open(
            "w", encoding="utf-8"
        )
        if generation_subset > 0:
            generation_handle = (output_path / "generation.jsonl").open(
                "w", encoding="utf-8"
            )

        for condition, bits in conditions:
            restore_linear_weights(modules, snapshot)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            started = time.perf_counter()
            if bits is None:
                inventory = fp_module_inventory(model)
            else:
                inventory = apply_rtn_to_model(model, bits, group_size)

            predictions = score_records(
                model=model,
                tokenizer=tokenizer,
                records=records,
                condition=condition,
                option_token_ids=option_token_ids,
                batch_size=int(config["scoring"]["batch_size"]),
                max_seq_len=int(config["scoring"]["max_seq_len"]),
            )
            _write_jsonl(output_path / f"predictions_{condition}.jsonl", predictions)
            _append_jsonl(predictions_handle, predictions)

            if generation_subset > 0:
                generation_rows = greedy_generation_subset(
                    model=model,
                    tokenizer=tokenizer,
                    records=records[:generation_subset],
                    condition=condition,
                    max_new_tokens=int(
                        config["generation"].get("max_new_tokens", 8)
                    ),
                    max_seq_len=int(config["scoring"]["max_seq_len"]),
                )
                _write_jsonl(
                    output_path / f"generation_{condition}.jsonl",
                    generation_rows,
                )
                _append_jsonl(generation_handle, generation_rows)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                peak_bytes: int | None = int(
                    torch.cuda.max_memory_allocated(device)
                )
            else:
                peak_bytes = None

            manifest["conditions"].append(
                {
                    "condition": condition,
                    "bits": bits,
                    "group_size": group_size if bits is not None else None,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_gpu_allocated_bytes": peak_bytes,
                    "module_inventory": inventory,
                }
            )
            manifest["completed_conditions"].append(condition)
            _write_json(output_path / "run.json", manifest)

        manifest["state"] = "completed"
        _write_json(output_path / "run.json", manifest)
    except Exception as error:
        manifest["state"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json(output_path / "run.json", manifest)
        raise
    finally:
        if modules is not None and snapshot is not None:
            restore_linear_weights(modules, snapshot)
        if predictions_handle is not None:
            predictions_handle.close()
        if generation_handle is not None:
            generation_handle.close()
    return manifest
