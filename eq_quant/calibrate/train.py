"""Single-objective calibration fit over frozen RTN codes."""

import copy
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eq_quant.evaluate.runner import load_model, load_records, set_determinism
from eq_quant.evaluate.scoring import (
    build_chat_prompt,
    encode_prompts,
    verify_option_token_ids,
)

from .basis import (
    build_rtn_basis,
    export_scales,
    quantized_weight_storage,
    replace_linear_modules,
    trainable_parameter_count,
    trainable_parameter_names,
)
from .io_helpers import (
    hardware_metadata,
    peak_memory_bytes,
    prepare_output_files,
    reset_peak_memory,
    software_metadata,
    write_json,
)
from .objectives import (
    BatchRow,
    build_batch_rows,
    canonical_option_distributions,
    cosine_gram_mse,
    full_vocabulary_kl,
    objective_is_regularized,
    paired_js,
    random_pair_js,
)
from .records import (
    CALIBRATION_SPLIT,
    select_calibration_pairs,
    validate_calibration_records,
)
from .teacher import load_teacher_cache

MANIFEST_FILENAME = "calibration.json"
SCALES_FILENAME = "scales.pt"


def calibration_settings(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("calibration")
    if not isinstance(block, dict):
        raise ValueError("config.calibration is required")
    split = str(block.get("split", CALIBRATION_SPLIT))
    if split != CALIBRATION_SPLIT:
        raise ValueError(
            "calibration fits require calibration.split == 'calibration', "
            f"got {split!r}"
        )
    settings = {
        "bits": int(block["bits"]),
        "group_size": int(block["group_size"]),
        "split": split,
        "learning_rate": float(block.get("learning_rate", 0.001)),
        "betas": tuple(float(value) for value in block.get("betas", (0.9, 0.999))),
        "weight_decay": float(block.get("weight_decay", 0.0)),
        "gradient_clip_norm": float(block.get("gradient_clip_norm", 1.0)),
        "bases_per_batch": int(block.get("bases_per_batch", 4)),
        "prompts_per_batch": int(block.get("prompts_per_batch", 8)),
        "total_updates": int(block.get("total_updates", 96)),
        "temperature": float(block.get("temperature", 1.0)),
        "gradient_checkpointing": bool(block.get("gradient_checkpointing", False)),
        "cpu_threads": int(block.get("cpu_threads", 6)),
    }
    quantization = config["quantization"]
    if settings["group_size"] != int(quantization["group_size"]):
        raise ValueError("calibration.group_size must match quantization.group_size")
    if settings["bits"] not in [int(value) for value in quantization["bits"]]:
        raise ValueError("calibration.bits must be one of quantization.bits")
    if settings["bases_per_batch"] < 1:
        raise ValueError("calibration.bases_per_batch must be positive")
    if settings["prompts_per_batch"] != 2 * settings["bases_per_batch"]:
        raise ValueError("calibration.prompts_per_batch must be twice bases_per_batch")
    if settings["total_updates"] < 1:
        raise ValueError("calibration.total_updates must be positive")
    if settings["temperature"] != 1.0:
        raise ValueError("the frozen distillation temperature is 1")
    if len(settings["betas"]) != 2:
        raise ValueError("calibration.betas must contain two values")
    return settings


def _student_forward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    need_hidden: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    if not torch.all(attention_mask[:, -1] == 1):
        raise ValueError("left padding requires a valid final token in every row")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        logits_to_keep=1,
        output_hidden_states=need_hidden,
    )
    if outputs.logits.dim() != 3 or outputs.logits.shape[1] != 1:
        raise ValueError("expected last-token logits with shape [batch, 1, vocab]")
    logits = outputs.logits[:, -1, :]
    hidden = None
    if need_hidden:
        if outputs.hidden_states is None:
            raise ValueError("model did not return hidden states")
        hidden = outputs.hidden_states[-1][:, -1, :]
    return logits, hidden


def _zero_dropout(model: nn.Module) -> None:
    """Disable every supported dropout path, including attention modules.

    Qwen/Llama attention layers cache ``config.attention_dropout`` on the
    module at construction time, so zeroing ``nn.Dropout`` modules and the
    config alone is not sufficient.
    """
    config = getattr(model, "config", None)
    if config is not None:
        for name in (
            "attention_dropout",
            "hidden_dropout",
            "activation_dropout",
            "dropout",
        ):
            if hasattr(config, name):
                setattr(config, name, 0.0)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        for name in ("attention_dropout", "dropout", "dropout_prob"):
            value = getattr(module, name, None)
            if isinstance(value, float):
                setattr(module, name, 0.0)


def activate_calibration_mode(
    model: nn.Module,
    gradient_checkpointing: bool,
) -> None:
    """Put the model in training mode with dropout fully disabled.

    HF gradient checkpointing is only active in training mode, so calibration
    always uses training mode even when checkpointing is off. Dropout is zero,
    which makes training-mode forward passes deterministic.
    """
    _zero_dropout(model)
    model.train()
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = False
    if gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise ValueError("model does not support gradient checkpointing")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


def _row_teacher_logits(
    payload: dict[str, Any],
    rows: list[BatchRow],
    device: torch.device,
) -> torch.Tensor:
    indices = [row.teacher_index for row in rows]
    return payload["logits"].index_select(
        0, torch.tensor(indices, dtype=torch.long)
    ).to(device=device, dtype=torch.float32)


def _row_teacher_hidden(
    payload: dict[str, Any],
    rows: list[BatchRow],
    device: torch.device,
) -> torch.Tensor:
    indices = [row.teacher_index for row in rows]
    return payload["hidden"].index_select(
        0, torch.tensor(indices, dtype=torch.long)
    ).to(device=device, dtype=torch.float32)


def run_calibration(
    config: dict[str, Any],
    records_path: str | Path,
    teacher_cache_path: str | Path,
    output_dir: str | Path,
    objective: str,
    lambda_value: float | None,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    settings = calibration_settings(config)
    regularized = objective_is_regularized(objective)
    if regularized:
        if lambda_value is None or lambda_value <= 0:
            raise ValueError(
                f"objective {objective!r} requires a positive explicit --lambda"
            )
    elif lambda_value is not None:
        raise ValueError(
            f"KD control {objective!r} has no regularization lambda"
        )

    records = load_records(records_path)
    validate_calibration_records(records, settings["split"])
    pairs = select_calibration_pairs(records, settings["split"])
    expected_bases = settings["bases_per_batch"] * settings["total_updates"]
    if len(pairs) != expected_bases:
        raise ValueError(
            f"calibration has {len(pairs)} base pairs but the frozen schedule "
            f"needs {expected_bases} "
            f"({settings['bases_per_batch']} bases x {settings['total_updates']} updates)"
        )

    teacher_payload = load_teacher_cache(
        teacher_cache_path,
        expected_records=[
            record for pair in pairs for record in pair.ordered_records()
        ],
        model_config=config["model"],
        scoring_config=config["scoring"],
    )
    teacher_index = {
        record_id: index
        for index, record_id in enumerate(teacher_payload["record_ids"])
    }

    output_path = prepare_output_files(
        output_dir, [MANIFEST_FILENAME, SCALES_FILENAME], overwrite
    )
    manifest: dict[str, Any] = {
        "state": "running",
        "objective": objective,
        "lambda": lambda_value,
        "fit_seed": int(seed),
        "config": copy.deepcopy(config),
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "dtype": config["model"]["dtype"],
            "resolved_commit_hash": None,
        },
        "quantization": {
            "method": "frozen_symmetric_signed_rtn_codes",
            "bits": settings["bits"],
            "group_size": settings["group_size"],
            "trainable": (
                "per-output-row/per-input-group relative log scale deltas; "
                "effective scale = fixed FP32 RTN scale * exp(delta)"
            ),
            "codes_frozen": True,
            "biases_embeddings_head_frozen": True,
            "forward_weight_dtype": "bfloat16",
            "zero_groups_remain_zero": True,
        },
        "records": {
            "path": str(records_path),
            "bases": len(pairs),
            "prompts": 2 * len(pairs),
        },
        "teacher_cache": {
            "path": str(teacher_cache_path),
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": settings["learning_rate"],
            "betas": list(settings["betas"]),
            "weight_decay": settings["weight_decay"],
            "gradient_clip_norm": settings["gradient_clip_norm"],
            "bases_per_batch": settings["bases_per_batch"],
            "prompts_per_batch": settings["prompts_per_batch"],
            "total_updates": settings["total_updates"],
            "epochs": 1,
            "temperature": settings["temperature"],
            "initialization": (
                "delta=0; effective scale = fixed FP32 RTN scale * exp(delta)"
            ),
            "scale_delta_initialized_zero": True,
            "rtn_scales_dtype": "float32",
            "dropout": 0.0,
            "gradient_checkpointing": settings["gradient_checkpointing"],
            "gradient_checkpointing_reentrant": False,
        },
        "trainable_parameters": None,
        "trainable_parameter_names": None,
        "pinned_code_counts": None,
        "steps": [],
        "token_counts": {
            "nonpadding_total": None,
            "padded_total": None,
            "per_update": [],
        },
        "batch_order": [],
        "elapsed_seconds": None,
        "optimization_elapsed_seconds": None,
        "peak_memory_bytes": None,
        "software": software_metadata(),
        "hardware": None,
        "error": None,
    }
    write_json(output_path / MANIFEST_FILENAME, manifest)

    model = None
    modules = None
    total_started = time.perf_counter()
    try:
        set_determinism(seed=seed, cpu_threads=settings["cpu_threads"])
        model, tokenizer, device = load_model(config)
        tokenizer.padding_side = "left"
        manifest["model"]["resolved_commit_hash"] = getattr(
            model.config, "_commit_hash", None
        )
        manifest["hardware"] = hardware_metadata(device)

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        basis = build_rtn_basis(model, settings["bits"], settings["group_size"])
        modules = replace_linear_modules(model, basis)
        activate_calibration_mode(model, settings["gradient_checkpointing"])

        trainable_names = trainable_parameter_names(model)
        if not trainable_names or any(
            not name.endswith("log_scale_deltas") for name in trainable_names
        ):
            raise RuntimeError(
                f"unexpected trainable parameters: {trainable_names}"
            )
        manifest["trainable_parameters"] = trainable_parameter_count(model)
        manifest["trainable_parameter_names"] = trainable_names
        manifest["pinned_code_counts"] = quantized_weight_storage(modules)

        option_token_ids = verify_option_token_ids(
            tokenizer,
            labels=tuple(config["scoring"]["option_labels"]),
            sample_context=build_chat_prompt(tokenizer, records[0]["prompt"]),
        )
        max_seq_len = int(config["scoring"]["max_seq_len"])

        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=settings["learning_rate"],
            betas=settings["betas"],
            weight_decay=settings["weight_decay"],
        )
        order = list(range(len(pairs)))
        random.Random(seed).shuffle(order)

        reset_peak_memory(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimization_started = time.perf_counter()
        total_nonpadding_tokens = 0
        total_padded_tokens = 0
        for update in range(settings["total_updates"]):
            start = update * settings["bases_per_batch"]
            end = start + settings["bases_per_batch"]
            batch_pairs = [pairs[index] for index in order[start:end]]
            rows = build_batch_rows(batch_pairs, objective, teacher_index)
            row_records = [row.record for row in rows]
            texts = [
                build_chat_prompt(tokenizer, record["prompt"])
                for record in row_records
            ]
            encoded = encode_prompts(tokenizer, texts, max_seq_len)
            update_padded_tokens = int(encoded["input_ids"].numel())
            update_nonpadding_tokens = int(encoded["attention_mask"].sum())
            total_padded_tokens += update_padded_tokens
            total_nonpadding_tokens += update_nonpadding_tokens

            need_hidden = objective == "kernel"
            student_logits, student_hidden = _student_forward(
                model,
                encoded["input_ids"],
                encoded["attention_mask"],
                need_hidden,
            )
            teacher_logits = _row_teacher_logits(teacher_payload, rows, device)
            kd_loss = full_vocabulary_kl(teacher_logits, student_logits)

            regularizer_value = torch.zeros((), device=device, dtype=torch.float32)
            if objective in ("paired", "random_pair", "kernel"):
                if objective == "kernel":
                    teacher_hidden = _row_teacher_hidden(
                        teacher_payload, rows, device
                    )
                    regularizer_value = cosine_gram_mse(
                        teacher_hidden, student_hidden
                    )
                else:
                    canonical = canonical_option_distributions(
                        student_logits, row_records, option_token_ids
                    )
                    if objective == "paired":
                        regularizer_value = paired_js(
                            canonical, len(batch_pairs)
                        )
                    else:
                        regularizer_value = random_pair_js(
                            canonical, len(batch_pairs)
                        )

            if regularized:
                total_loss = kd_loss + float(lambda_value) * regularizer_value
            else:
                total_loss = kd_loss

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"nonfinite total loss at update {update}: "
                    f"{float(total_loss.detach().cpu())}"
                )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                settings["gradient_clip_norm"],
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"nonfinite gradient norm at update {update}: "
                    f"{float(gradient_norm.detach().cpu())}"
                )
            optimizer.step()

            manifest["steps"].append(
                {
                    "update": update,
                    "kd_loss": float(kd_loss.detach().cpu()),
                    "regularizer_loss": float(
                        regularizer_value.detach().cpu()
                    ),
                    "total_loss": float(total_loss.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "nonpadding_tokens": update_nonpadding_tokens,
                    "padded_tokens": update_padded_tokens,
                }
            )
            manifest["token_counts"]["per_update"].append(
                {
                    "update": update,
                    "nonpadding": update_nonpadding_tokens,
                    "padded": update_padded_tokens,
                }
            )
            manifest["batch_order"].append(
                {
                    "update": update,
                    "base_ids": [pair.base_id for pair in batch_pairs],
                    "record_ids": [
                        record["record_id"] for record in row_records
                    ],
                }
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimization_elapsed = time.perf_counter() - optimization_started
        manifest["token_counts"]["nonpadding_total"] = total_nonpadding_tokens
        manifest["token_counts"]["padded_total"] = total_padded_tokens
        manifest["optimization_elapsed_seconds"] = optimization_elapsed
        manifest["peak_memory_bytes"] = peak_memory_bytes(device)

        scales = export_scales(modules)
        for name, values in scales.items():
            if not torch.isfinite(values).all():
                raise FloatingPointError(
                    f"nonfinite exported scale deltas for {name}"
                )
        torch.save(
            {
                "format": "eq_quant_calibration_scales_v1",
                "scale_parameterization": "rtn_group_scale_times_exp_delta",
                "scale_deltas": scales,
                "rtn_scale_reference": (
                    "fixed FP32 per-group RTN scale rebuilt deterministically "
                    "from the pinned model revision and quantization settings"
                ),
                "bits": settings["bits"],
                "group_size": settings["group_size"],
                "model": {
                    "id": config["model"]["id"],
                    "revision": config["model"]["revision"],
                },
                "objective": objective,
                "lambda": lambda_value,
                "fit_seed": int(seed),
            },
            output_path / SCALES_FILENAME,
        )

        manifest["state"] = "completed"
        manifest["elapsed_seconds"] = time.perf_counter() - total_started
        write_json(output_path / MANIFEST_FILENAME, manifest)
    except Exception as error:
        manifest["state"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        if model is not None:
            manifest["peak_memory_bytes"] = peak_memory_bytes(
                next(model.parameters()).device
            )
        manifest["elapsed_seconds"] = time.perf_counter() - total_started
        write_json(output_path / MANIFEST_FILENAME, manifest)
        raise
    return manifest
