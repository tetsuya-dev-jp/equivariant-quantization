"""Frozen RTN codes with trainable relative logarithmic output scales.

The integer codes and all pretrained parameters stay frozen. The effective
scale of each (output row, input group) is

    s = s_rtn * exp(delta)

where ``s_rtn`` is the fixed FP32 round-to-nearest scale and ``delta`` is the
only trainable tensor, initialized to exact zero. Multiplying from a stable
origin instead of exponentiating a stored logarithm means the initial weight
reconstruction matches the RTN BF16 dequantized weight exactly. Zero RTN groups
keep ``s = 0`` for every delta value.

Only the relative deltas are exported. The fixed ``s_rtn`` buffers are rebuilt
deterministically from the pinned model revision and quantization settings, so
an exported delta restores the identical effective scale.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from eq_quant.evaluate.quantize import quantizable_linear_modules, quantize_weight_rtn


@dataclass(frozen=True)
class ModuleBasis:
    """Pinned quantizer state for one quantizable linear module."""

    name: str
    codes: torch.Tensor
    rtn_scales: torch.Tensor
    zero_groups: torch.Tensor
    group_size: int
    in_features: int
    out_features: int
    bias: torch.Tensor | None

    @property
    def groups(self) -> int:
        return self.rtn_scales.shape[1]


def _padded_codes(codes: torch.Tensor, padded_features: int) -> torch.Tensor:
    out_features, in_features = codes.shape
    padded = torch.zeros(
        (out_features, padded_features), dtype=torch.int8, device=codes.device
    )
    padded[:, :in_features] = codes
    return padded


def build_rtn_basis(
    model: nn.Module,
    bits: int,
    group_size: int,
) -> list[ModuleBasis]:
    """Quantize every eligible linear weight once and keep its integer codes."""
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    basis: list[ModuleBasis] = []
    for name, module in quantizable_linear_modules(model):
        result = quantize_weight_rtn(module.weight, bits, group_size)
        groups = int(result.scales.shape[1])
        out_features, in_features = module.weight.shape
        padded_features = groups * group_size

        # Keep the exact FP32 RTN scale as a fixed buffer. The trainable
        # quantity is a relative log delta initialized to zero.
        rtn_scales = result.scales.to(torch.float32).contiguous()
        zero_groups = rtn_scales <= 0

        bias = None
        if module.bias is not None:
            bias = module.bias.detach().clone()

        basis.append(
            ModuleBasis(
                name=name,
                codes=_padded_codes(result.codes, padded_features),
                rtn_scales=rtn_scales,
                zero_groups=zero_groups.contiguous(),
                group_size=group_size,
                in_features=int(in_features),
                out_features=int(out_features),
                bias=bias,
            )
        )
    return basis


class CalibratedLinear(nn.Module):
    """Linear layer rebuilt from frozen codes and relative log scale deltas."""

    def __init__(self, basis: ModuleBasis) -> None:
        super().__init__()
        self.in_features = basis.in_features
        self.out_features = basis.out_features
        self.group_size = basis.group_size
        self.padded_features = basis.groups * basis.group_size

        self.register_buffer("codes", basis.codes, persistent=True)
        self.register_buffer("rtn_scales", basis.rtn_scales, persistent=True)
        self.register_buffer("zero_groups", basis.zero_groups, persistent=True)
        if basis.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", basis.bias, persistent=True)

        # Effective scale = rtn_scales * exp(log_scale_deltas).
        self.log_scale_deltas = nn.Parameter(
            torch.zeros_like(basis.rtn_scales)
        )

    def effective_scales(self) -> torch.Tensor:
        scales = self.rtn_scales * torch.exp(self.log_scale_deltas)
        return torch.where(
            self.zero_groups, torch.zeros_like(scales), scales
        )

    def reconstructed_weight(self, dtype: torch.dtype) -> torch.Tensor:
        scales = self.effective_scales()
        expanded = (
            scales.unsqueeze(-1)
            .expand(-1, -1, self.group_size)
            .reshape(self.out_features, self.padded_features)
        )
        weight = self.codes.to(torch.float32) * expanded
        weight = weight[:, : self.in_features]
        # The plan reconstructs BF16 weights during forward passes. When the
        # surrounding model computes in another dtype (for example a float32
        # test model), cast after BF16 rounding so ordinary modules still run.
        return weight.to(torch.bfloat16).to(dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.reconstructed_weight(hidden_states.dtype)
        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype=hidden_states.dtype)
        return F.linear(hidden_states, weight, bias)


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_path, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, child_name, module)


def replace_linear_modules(
    model: nn.Module,
    basis: list[ModuleBasis],
) -> dict[str, CalibratedLinear]:
    """Replace each quantizable linear module in place."""
    modules: dict[str, CalibratedLinear] = {}
    for entry in basis:
        wrapper = CalibratedLinear(entry)
        _set_submodule(model, entry.name, wrapper)
        modules[entry.name] = wrapper
    return modules


def export_scales(modules: dict[str, CalibratedLinear]) -> dict[str, torch.Tensor]:
    """Return detached FP32 CPU relative log deltas for every module."""
    return {
        name: module.log_scale_deltas.detach().to("cpu", torch.float32).clone()
        for name, module in modules.items()
    }


def load_scales(
    modules: dict[str, CalibratedLinear],
    log_scale_deltas: dict[str, torch.Tensor],
) -> None:
    """Copy exported relative log deltas into an identically built model."""
    if set(modules) != set(log_scale_deltas):
        missing = sorted(set(modules) - set(log_scale_deltas))
        unexpected = sorted(set(log_scale_deltas) - set(modules))
        raise ValueError(
            "exported scale module names do not match the model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, module in modules.items():
            values = log_scale_deltas[name]
            if tuple(values.shape) != tuple(module.log_scale_deltas.shape):
                raise ValueError(
                    f"scale shape mismatch for {name}: "
                    f"{tuple(values.shape)} != {tuple(module.log_scale_deltas.shape)}"
                )
            module.log_scale_deltas.copy_(
                values.to(
                    module.log_scale_deltas.device,
                    module.log_scale_deltas.dtype,
                )
            )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]


def quantized_weight_storage(modules: dict[str, CalibratedLinear]) -> dict[str, int]:
    """Counts of pinned integer codes and FP32 learnable deltas."""
    return {
        "codes": sum(module.codes.numel() for module in modules.values()),
        "scales": sum(module.log_scale_deltas.numel() for module in modules.values()),
        "zero_groups": sum(int(module.zero_groups.sum()) for module in modules.values()),
    }
