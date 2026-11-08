"""Frozen calibration objectives and their batch layouts."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

OBJECTIVES = (
    "original_kd",
    "augmented_kd",
    "paired",
    "random_pair",
    "kernel",
)

REGULARIZED_OBJECTIVES = ("paired", "random_pair", "kernel")


@dataclass(frozen=True)
class BatchRow:
    record: dict[str, Any]
    teacher_index: int


def objective_is_regularized(objective: str) -> bool:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}")
    return objective in REGULARIZED_OBJECTIVES


def build_batch_rows(
    pairs: list[Any],
    objective: str,
    teacher_index: dict[str, int],
) -> list[BatchRow]:
    """Expand four calibration bases into the objective's eight prompt rows."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}")
    rows: list[BatchRow] = []
    for pair in pairs:
        original = pair.original
        variant = pair.option_permutation
        rows.append(BatchRow(original, teacher_index[original["record_id"]]))
        if objective == "original_kd":
            # Match prompt count and updates by duplicating originals.
            rows.append(BatchRow(original, teacher_index[original["record_id"]]))
        else:
            rows.append(BatchRow(variant, teacher_index[variant["record_id"]]))
    return rows


def full_vocabulary_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(teacher || student) at temperature 1, averaged over prompts."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher and student logits must match: "
            f"{tuple(teacher_logits.shape)} != {tuple(student_logits.shape)}"
        )
    teacher_log_probabilities = torch.log_softmax(
        teacher_logits.to(torch.float32), dim=-1
    )
    student_log_probabilities = torch.log_softmax(
        student_logits.to(torch.float32), dim=-1
    )
    teacher_probabilities = teacher_log_probabilities.exp()
    per_row = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1)
    return per_row.mean()


def _inverse_option_order(option_order: list[int], device: torch.device) -> torch.Tensor:
    inverse = torch.empty(len(option_order), dtype=torch.long, device=device)
    for display_index, canonical_index in enumerate(option_order):
        inverse[canonical_index] = display_index
    return inverse


def canonical_option_distributions(
    student_logits: torch.Tensor,
    records: list[dict[str, Any]],
    option_token_ids: list[int],
) -> torch.Tensor:
    """Map each row's four displayed option probabilities to canonical order."""
    if len(records) != student_logits.shape[0]:
        raise ValueError("one record is required per student row")
    option_ids = torch.tensor(
        option_token_ids, dtype=torch.long, device=student_logits.device
    )
    display_logits = student_logits.index_select(dim=1, index=option_ids)
    display_probabilities = torch.softmax(display_logits.to(torch.float32), dim=-1)

    rows: list[torch.Tensor] = []
    for row, record in enumerate(records):
        option_order = list(record["option_order"])
        if sorted(option_order) != list(range(len(option_order))):
            raise ValueError(
                f"record {record['record_id']} has invalid option_order {option_order}"
            )
        inverse = _inverse_option_order(option_order, display_probabilities.device)
        rows.append(display_probabilities[row].gather(0, inverse))
    return torch.stack(rows, dim=0)


def _kl_against_midpoint(
    probabilities: torch.Tensor,
    midpoint: torch.Tensor,
) -> torch.Tensor:
    """KL(probabilities || midpoint) with finite gradients at exact zeros.

    ``torch.xlogy`` has a nonfinite backward at ``(0, 0)``. Degenerate
    distributions can therefore yield a finite forward JS value with NaN
    gradients. Masking the zero-probability terms and replacing every log
    argument by one avoids evaluating ``log(0)`` in the backward pass.
    """
    positive = probabilities > 0
    safe_probabilities = torch.where(
        positive, probabilities, torch.ones_like(probabilities)
    )
    safe_midpoint = torch.where(
        midpoint > 0, midpoint, torch.ones_like(midpoint)
    )
    terms = safe_probabilities * (
        torch.log(safe_probabilities) - torch.log(safe_midpoint)
    )
    terms = torch.where(positive, terms, torch.zeros_like(terms))
    return terms.sum(dim=-1)


def jensen_shannon(probabilities: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence in FP32 probability space, per row."""
    if probabilities.shape != other.shape:
        raise ValueError("JS inputs must have the same shape")
    probabilities = probabilities.to(torch.float32)
    other = other.to(torch.float32)
    midpoint = 0.5 * (probabilities + other)
    first = _kl_against_midpoint(probabilities, midpoint)
    second = _kl_against_midpoint(other, midpoint)
    return 0.5 * (first + second)


def paired_js(
    canonical_probabilities: torch.Tensor,
    num_bases: int,
) -> torch.Tensor:
    """JS between each base's original and option-permuted distributions."""
    if canonical_probabilities.shape[0] != 2 * num_bases:
        raise ValueError("paired layout requires exactly two rows per base")
    originals = canonical_probabilities[0::2]
    variants = canonical_probabilities[1::2]
    return jensen_shannon(originals, variants).mean()


def random_pair_js(
    canonical_probabilities: torch.Tensor,
    num_bases: int,
) -> torch.Tensor:
    """JS matching each original to another base's variant via a cyclic shift."""
    if num_bases < 2:
        raise ValueError("random-pair control requires at least two bases")
    if canonical_probabilities.shape[0] != 2 * num_bases:
        raise ValueError("augmented layout requires exactly two rows per base")
    originals = canonical_probabilities[0::2]
    variants = canonical_probabilities[1::2]
    shift = torch.arange(1, num_bases + 1, device=variants.device) % num_bases
    shifted_variants = variants.index_select(0, shift)
    return jensen_shannon(originals, shifted_variants).mean()


def cosine_gram_mse(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
) -> torch.Tensor:
    """Mean squared difference of cosine Gram matrices in one minibatch."""
    if teacher_hidden.shape != student_hidden.shape:
        raise ValueError(
            "teacher and student hidden shapes must match: "
            f"{tuple(teacher_hidden.shape)} != {tuple(student_hidden.shape)}"
        )
    teacher_normalized = F.normalize(
        teacher_hidden.to(torch.float32), dim=-1, eps=1e-8
    )
    student_normalized = F.normalize(
        student_hidden.to(torch.float32), dim=-1, eps=1e-8
    )
    teacher_gram = teacher_normalized @ teacher_normalized.transpose(0, 1)
    student_gram = student_normalized @ student_normalized.transpose(0, 1)
    return ((teacher_gram - student_gram) ** 2).mean()
