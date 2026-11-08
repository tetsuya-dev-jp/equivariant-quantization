"""JSON configuration loading and strict validation."""

import json
from pathlib import Path
from typing import Any

from .data.schema import DEFAULT_SPLIT_SPECS, FAMILIES, OPTION_LABELS, SPLIT_ORDER


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("config.model is required")
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("config.model.id is required")
    revision = model.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError(
            "config.model.revision is required; pin an immutable model revision"
        )
    if model.get("dtype") not in {"bfloat16", "float16"}:
        raise ValueError("config.model.dtype must be 'bfloat16' or 'float16'")

    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("config.quantization is required")
    group_size = quantization.get("group_size")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("config.quantization.group_size must be a positive integer")
    bits = quantization.get("bits")
    if not isinstance(bits, list) or not bits:
        raise ValueError("config.quantization.bits must be a non-empty list")
    for bit_width in bits:
        if not isinstance(bit_width, int) or not 2 <= bit_width <= 8:
            raise ValueError(f"invalid bit width: {bit_width!r}")

    scoring = config.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("config.scoring is required")
    batch_size = scoring.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("config.scoring.batch_size must be a positive integer")
    max_seq_len = scoring.get("max_seq_len")
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("config.scoring.max_seq_len must be a positive integer")
    labels = scoring.get("option_labels", list(OPTION_LABELS))
    if tuple(labels) != OPTION_LABELS:
        raise ValueError(f"config.scoring.option_labels must be {list(OPTION_LABELS)}")

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("config.data is required")
    families = data.get("families", list(FAMILIES))
    if not isinstance(families, list) or not families:
        raise ValueError("config.data.families must be a non-empty list")
    unknown_families = [family for family in families if family not in FAMILIES]
    if unknown_families:
        raise ValueError(f"unknown task families: {unknown_families}")

    splits = data.get("splits", DEFAULT_SPLIT_SPECS)
    if not isinstance(splits, dict):
        raise ValueError("config.data.splits must be an object")
    for split in SPLIT_ORDER:
        spec = splits.get(split, DEFAULT_SPLIT_SPECS[split])
        if not isinstance(spec.get("count"), int) or spec["count"] < 0:
            raise ValueError(f"config.data.splits.{split}.count is invalid")
        if not isinstance(spec.get("seed"), int):
            raise ValueError(f"config.data.splits.{split}.seed must be an integer")
