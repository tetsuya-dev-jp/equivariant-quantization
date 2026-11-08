"""Complete last-token teacher cache over calibration prompts."""

import time
from pathlib import Path
from typing import Any

import torch

from eq_quant.evaluate.runner import load_model, load_records, set_determinism
from eq_quant.evaluate.scoring import build_chat_prompt, encode_prompts

from .io_helpers import (
    hardware_metadata,
    peak_memory_bytes,
    prepare_output_files,
    reset_peak_memory,
    software_metadata,
    write_json,
)
from .records import (
    CALIBRATION_SPLIT,
    ordered_record_metadata,
    record_metadata,
    select_calibration_pairs,
    validate_calibration_records,
)

CACHE_FILENAME = "teacher_cache.pt"
MANIFEST_FILENAME = "cache_manifest.json"


def _teacher_batch(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    if not torch.all(attention_mask[:, -1] == 1):
        raise ValueError("left padding requires a valid final token in every row")

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
            output_hidden_states=True,
        )
    if outputs.logits.dim() != 3 or outputs.logits.shape[1] != 1:
        raise ValueError("expected last-token logits with shape [batch, 1, vocab]")
    if outputs.hidden_states is None:
        raise ValueError("model did not return hidden states")
    logits = outputs.logits[:, -1, :]
    hidden = outputs.hidden_states[-1][:, -1, :]
    return logits.detach().to("cpu"), hidden.detach().to("cpu")


def _require_calibration_config(config: dict[str, Any]) -> None:
    block = config.get("calibration")
    if isinstance(block, dict):
        configured = str(block.get("split", CALIBRATION_SPLIT))
        if configured != CALIBRATION_SPLIT:
            raise ValueError(
                "teacher caching requires calibration.split == 'calibration', "
                f"got {configured!r}"
            )


def cache_teacher(
    config: dict[str, Any],
    records_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    _require_calibration_config(config)
    split = CALIBRATION_SPLIT
    all_records = load_records(records_path)
    # The cache stores calibration prompts only, even if the input file also
    # contains other splits.
    records = [
        record for record in all_records if record.get("split") == split
    ]
    validate_calibration_records(records, split)
    pairs = select_calibration_pairs(records, split)
    metadata = ordered_record_metadata(pairs)
    record_ids = [entry["record_id"] for entry in metadata]

    output_path = prepare_output_files(
        output_dir, [CACHE_FILENAME, MANIFEST_FILENAME], overwrite
    )
    manifest: dict[str, Any] = {
        "state": "running",
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "dtype": config["model"]["dtype"],
            "resolved_commit_hash": None,
        },
        "scoring": {
            "max_seq_len": int(config["scoring"]["max_seq_len"]),
            "option_labels": list(config["scoring"]["option_labels"]),
        },
        "records": {
            "path": str(records_path),
            "count": len(records),
            "bases": len(pairs),
            "transforms": ["original", "option_permutation"],
        },
        "cache": {
            "file": CACHE_FILENAME,
            "contents": [
                "complete final next-token vocabulary logits",
                "final normalized last hidden vector",
            ],
        },
        "software": software_metadata(),
        "hardware": None,
        "token_counts": {"total": 0},
        "elapsed_seconds": None,
        "peak_memory_bytes": None,
        "error": None,
    }
    write_json(output_path / MANIFEST_FILENAME, manifest)

    model = None
    started = time.perf_counter()
    try:
        set_determinism(
            seed=int(config.get("analysis", {}).get("seed", 4108)),
            cpu_threads=int(config["scoring"].get("cpu_threads", 6)),
        )
        model, tokenizer, device = load_model(config)
        tokenizer.padding_side = "left"
        reset_peak_memory(device)
        manifest["hardware"] = hardware_metadata(device)
        manifest["model"]["resolved_commit_hash"] = getattr(
            model.config, "_commit_hash", None
        )

        batch_size = int(config["scoring"]["batch_size"])
        max_seq_len = int(config["scoring"]["max_seq_len"])
        prompt_texts = [
            build_chat_prompt(tokenizer, record["prompt"])
            for record in records
        ]

        logits_rows: list[torch.Tensor] = []
        hidden_rows: list[torch.Tensor] = []
        token_counts: list[int] = []
        for start in range(0, len(records), batch_size):
            batch_texts = prompt_texts[start : start + batch_size]
            encoded = encode_prompts(tokenizer, batch_texts, max_seq_len)
            batch_logits, batch_hidden = _teacher_batch(
                model, encoded["input_ids"], encoded["attention_mask"]
            )
            logits_rows.append(batch_logits)
            hidden_rows.append(batch_hidden)
            token_counts.extend(
                int(count) for count in encoded["attention_mask"].sum(dim=1).tolist()
            )

        logits = torch.cat(logits_rows, dim=0)
        hidden = torch.cat(hidden_rows, dim=0)
        token_count_tensor = torch.tensor(token_counts, dtype=torch.long)

        torch.save(
            {
                "format": "eq_quant_teacher_cache_v1",
                "model": {
                    "id": config["model"]["id"],
                    "revision": config["model"]["revision"],
                    "dtype": config["model"]["dtype"],
                },
                "scoring": {
                    "max_seq_len": max_seq_len,
                    "option_labels": list(config["scoring"]["option_labels"]),
                },
                "records": metadata,
                "record_ids": record_ids,
                "token_counts": token_count_tensor,
                "logits": logits,
                "hidden": hidden,
            },
            output_path / CACHE_FILENAME,
        )

        manifest["state"] = "completed"
        manifest["token_counts"] = {
            "total": int(token_count_tensor.sum()),
            "per_record": token_counts,
        }
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["peak_memory_bytes"] = peak_memory_bytes(device)
        write_json(output_path / MANIFEST_FILENAME, manifest)
    except Exception as error:
        manifest["state"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["elapsed_seconds"] = time.perf_counter() - started
        if model is not None:
            manifest["peak_memory_bytes"] = peak_memory_bytes(
                next(model.parameters()).device
            )
        write_json(output_path / MANIFEST_FILENAME, manifest)
        raise
    return manifest


def _first_metadata_mismatch(
    cached: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> str:
    common = min(len(cached), len(expected))
    for index in range(common):
        if cached[index] != expected[index]:
            record_id = expected[index].get("record_id")
            return (
                f"record metadata mismatch at ordered index {index} "
                f"(record_id={record_id!r})"
            )
    return f"record metadata count mismatch: {len(cached)} != {len(expected)}"


def load_teacher_cache(
    cache_path: str | Path,
    expected_records: list[dict[str, Any]],
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    """Load a cache and reject any record or provenance mismatch directly."""
    path = Path(cache_path)
    if path.is_dir():
        path = path / CACHE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"teacher cache not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "eq_quant_teacher_cache_v1":
        raise ValueError(f"unrecognised teacher cache format in {path}")

    expected_metadata = [record_metadata(record) for record in expected_records]
    cached_metadata = list(payload.get("records", []))
    if cached_metadata != expected_metadata:
        raise ValueError(
            "teacher cache input records do not match the selected calibration "
            f"records: {_first_metadata_mismatch(cached_metadata, expected_metadata)}"
        )

    cached_model = payload.get("model", {})
    if cached_model.get("id") != model_config["id"]:
        raise ValueError(
            "teacher cache model id mismatch: "
            f"{cached_model.get('id')!r} != {model_config['id']!r}"
        )
    if cached_model.get("revision") != model_config["revision"]:
        raise ValueError(
            "teacher cache model revision mismatch: "
            f"{cached_model.get('revision')!r} != {model_config['revision']!r}"
        )
    if cached_model.get("dtype") != model_config["dtype"]:
        raise ValueError(
            "teacher cache model dtype mismatch: "
            f"{cached_model.get('dtype')!r} != {model_config['dtype']!r}"
        )

    cached_scoring = payload.get("scoring", {})
    expected_max_seq_len = int(scoring_config["max_seq_len"])
    if cached_scoring.get("max_seq_len") != expected_max_seq_len:
        raise ValueError(
            "teacher cache max_seq_len mismatch: "
            f"{cached_scoring.get('max_seq_len')!r} != {expected_max_seq_len!r}"
        )
    expected_labels = list(scoring_config["option_labels"])
    if list(cached_scoring.get("option_labels", [])) != expected_labels:
        raise ValueError(
            "teacher cache option_labels mismatch: "
            f"{cached_scoring.get('option_labels')!r} != {expected_labels!r}"
        )

    logits = payload["logits"]
    hidden = payload["hidden"]
    token_counts = payload["token_counts"]
    expected_rows = len(expected_metadata)
    if logits.shape[0] != expected_rows:
        raise ValueError("teacher cache logits row count is inconsistent")
    if hidden.shape[0] != expected_rows:
        raise ValueError("teacher cache hidden row count is inconsistent")
    if token_counts.shape[0] != expected_rows:
        raise ValueError("teacher cache token count length is inconsistent")
    return payload
