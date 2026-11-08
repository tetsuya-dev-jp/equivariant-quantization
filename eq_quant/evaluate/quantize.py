"""Symmetric signed round-to-nearest weight quantization.

The dequantized weights are used for BF16/FP16 arithmetic. This is a numerical
weight-only simulation of quantization error. It does not establish packed
integer storage, deployment parity, or inference speed benefits.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class RTNResult:
    """Quantized codes, per-group scales, and simulated dequantized weights."""

    dequantized: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor
    bits: int
    group_size: int


def quantize_weight_rtn(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 128,
) -> RTNResult:
    """Quantize a two-dimensional weight matrix along its input columns."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    if not 2 <= bits <= 8:
        raise ValueError(f"bits must be between 2 and 8, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    out_features, in_features = weight.shape
    num_groups = math.ceil(in_features / group_size)
    padded_features = num_groups * group_size

    work = weight.detach().to(torch.float32)
    if padded_features > in_features:
        work = torch.nn.functional.pad(work, (0, padded_features - in_features))
    grouped = work.reshape(out_features, num_groups, group_size)

    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    scales = absmax / qmax

    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.round(grouped / safe_scales)
    codes = codes.clamp(qmin, qmax)
    # Zero groups have scale zero and must dequantize to exact zero.
    codes = torch.where(scales > 0, codes, torch.zeros_like(codes))

    dequantized = codes * scales
    dequantized = dequantized.reshape(out_features, padded_features)[
        :, :in_features
    ]
    codes = codes.reshape(out_features, padded_features)[:, :in_features]

    return RTNResult(
        dequantized=dequantized.to(dtype=weight.dtype),
        codes=codes.to(torch.int8),
        scales=scales.squeeze(-1).contiguous(),
        bits=bits,
        group_size=group_size,
    )


def _is_excluded_module(name: str, module: nn.Module) -> bool:
    tail = name.split(".")[-1]
    if tail == "lm_head":
        return True
    if isinstance(module, nn.Embedding):
        return True
    if "embed" in tail or "embedding" in tail:
        return True
    return False


def quantizable_linear_modules(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Transformer linear weights only, excluding the head and embeddings."""
    modules: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_excluded_module(name, module):
            continue
        modules.append((name, module))
    return modules


def snapshot_linear_weights(
    modules: list[tuple[str, nn.Linear]],
) -> dict[str, torch.Tensor]:
    return {
        name: module.weight.detach().clone()
        for name, module in modules
    }


def restore_linear_weights(
    modules: list[tuple[str, nn.Linear]],
    snapshot: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, module in modules:
            module.weight.copy_(snapshot[name])


def _module_inventory_entry(
    name: str,
    module: nn.Linear,
    result: RTNResult,
) -> dict[str, Any]:
    code_bytes = math.ceil(result.codes.numel() * result.bits / 8)
    scale_bytes = result.scales.numel() * 4
    return {
        "name": name,
        "shape": list(module.weight.shape),
        "numel": int(module.weight.numel()),
        "bits": result.bits,
        "group_size": result.group_size,
        "groups": int(result.scales.shape[1]),
        "code_min": int(result.codes.min()),
        "code_max": int(result.codes.max()),
        "zero_groups": int((result.scales == 0).sum()),
        "simulation_code_bytes_lower_bound": code_bytes,
        "simulation_scale_bytes_float32": scale_bytes,
    }


def apply_rtn_to_model(
    model: nn.Module,
    bits: int,
    group_size: int = 128,
) -> list[dict[str, Any]]:
    """Replace quantizable weights by their dequantized simulation values."""
    modules = quantizable_linear_modules(model)
    inventory: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, module in modules:
            result = quantize_weight_rtn(module.weight, bits, group_size)
            module.weight.copy_(result.dequantized)
            inventory.append(_module_inventory_entry(name, module, result))
    return inventory


def fp_module_inventory(model: nn.Module) -> list[dict[str, Any]]:
    modules = quantizable_linear_modules(model)
    return [
        {
            "name": name,
            "shape": list(module.weight.shape),
            "numel": int(module.weight.numel()),
            "bits": None,
            "group_size": None,
            "groups": None,
            "code_min": None,
            "code_max": None,
            "zero_groups": None,
            "simulation_code_bytes_lower_bound": None,
            "simulation_scale_bytes_float32": None,
        }
        for name, module in modules
    ]
