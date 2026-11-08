"""Next-token multiple-choice scoring with the official chat template."""

import re
from typing import Any

import torch

from eq_quant.data.oracles import canonical_answer
from eq_quant.data.schema import OPTION_LABELS


def build_chat_prompt(tokenizer: Any, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def verify_option_token_ids(
    tokenizer: Any,
    labels: tuple[str, ...] = OPTION_LABELS,
    sample_context: str | None = None,
) -> list[int]:
    """Verify each option label is a single token in the scoring context."""
    token_ids: list[int] = []
    for label in labels:
        direct = tokenizer.encode(label, add_special_tokens=False)
        spaced = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(direct) != 1:
            raise ValueError(f"option label {label!r} is not one token: {direct}")
        if len(spaced) != 1:
            raise ValueError(f"option label {label!r} with space is not one token")
        token_ids.append(direct[0])

        if sample_context is not None:
            prefix = tokenizer.encode(sample_context, add_special_tokens=False)
            appended = tokenizer.encode(
                sample_context + label, add_special_tokens=False
            )
            if appended != [*prefix, direct[0]]:
                raise ValueError(
                    f"option label {label!r} is not a clean next token "
                    "after the chat context"
                )

    if len(set(token_ids)) != len(labels):
        raise ValueError(f"option label token ids are not distinct: {token_ids}")
    return token_ids


def encode_prompts(
    tokenizer: Any,
    prompt_texts: list[str],
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[1] > max_seq_len:
        raise ValueError(
            "prompt batch exceeds max_seq_len: "
            f"{encoded['input_ids'].shape[1]} > {max_seq_len}"
        )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def _last_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.dim() != 3 or logits.shape[1] != 1:
        raise ValueError(
            "expected logits_to_keep=1 output with shape [batch, 1, vocab]"
        )
    if not torch.all(attention_mask[:, -1] == 1):
        raise ValueError(
            "left padding requires a valid final token in every row"
        )
    return logits[:, -1, :]


def score_encoded_batch(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    option_token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score one left-padded batch; the caller controls batching."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    option_ids = torch.tensor(option_token_ids, device=device, dtype=torch.long)

    # Attention-derived positions make batched left-padded rows match the same
    # rows scored individually without padding.
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.clamp(min=0)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
        )
        last_logits = _last_logits(outputs.logits, attention_mask)
        option_logits = last_logits.index_select(dim=1, index=option_ids)
        # BF16 softmax rounds near-ties into exact ties; score in FP32 while
        # keeping the raw option logits for prediction.
        option_probabilities = torch.softmax(option_logits.float(), dim=-1)

    token_counts = attention_mask.sum(dim=1)
    return option_logits, option_probabilities, token_counts


def _canonicalize_option_probabilities(
    display_probabilities: list[float],
    option_order: list[int],
) -> list[float]:
    canonical = [0.0] * len(display_probabilities)
    for display_index, canonical_index in enumerate(option_order):
        canonical[canonical_index] = display_probabilities[display_index]
    return canonical


def score_records(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    condition: str,
    option_token_ids: list[int],
    batch_size: int,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    prompt_texts = [build_chat_prompt(tokenizer, record["prompt"]) for record in records]
    results: list[dict[str, Any]] = []

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch_texts = prompt_texts[start : start + batch_size]
        encoded = encode_prompts(tokenizer, batch_texts, max_seq_len)
        option_logits, option_probabilities, token_counts = score_encoded_batch(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            option_token_ids,
        )

        for row, record in enumerate(batch_records):
            option_order = list(record["option_order"])
            display_probabilities = option_probabilities[row].tolist()
            prediction_display = int(torch.argmax(option_logits[row]).item())
            canonical_prediction = option_order[prediction_display]
            canonical_probabilities = _canonicalize_option_probabilities(
                display_probabilities, option_order
            )

            correct_canonical = canonical_answer(
                record["task_family"],
                record["semantic"]["structure"],
                record["semantic"]["canonical_options"],
            )
            recomputed_correct_display = option_order.index(correct_canonical)
            if recomputed_correct_display != record["correct_option_index"]:
                raise ValueError(
                    f"record {record['record_id']} has an incorrect saved "
                    "correct_option_index"
                )

            results.append(
                {
                    "record_id": record["record_id"],
                    "base_id": record["base_id"],
                    "split": record["split"],
                    "task_family": record["task_family"],
                    "transform": record["transform"],
                    "condition": condition,
                    "semantic": record["semantic"],
                    "option_order": option_order,
                    "labels": list(OPTION_LABELS),
                    "option_token_ids": list(option_token_ids),
                    "option_logits": option_logits[row].tolist(),
                    "option_probabilities": display_probabilities,
                    "canonical_probabilities": canonical_probabilities,
                    "prediction_display": prediction_display,
                    "canonical_prediction": canonical_prediction,
                    "correct_option_index": record["correct_option_index"],
                    "correct": canonical_prediction == correct_canonical,
                    "token_count": int(token_counts[row].item()),
                }
            )
    return results


def parse_option_letter(text: str) -> str | None:
    """Accept only a bare option letter with limited surrounding punctuation."""
    match = re.fullmatch(
        r"(?:([A-Da-d])[.):,]?|\(([A-Da-d])\)[.:,]?)",
        text.strip(),
    )
    if match is None:
        return None
    return (match.group(1) or match.group(2)).upper()


def greedy_generation_subset(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    condition: str,
    max_new_tokens: int,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    """Greedy raw generation with explicit parse failures retained."""
    if not records:
        return []

    prompt_texts = [build_chat_prompt(tokenizer, record["prompt"]) for record in records]
    encoded = encode_prompts(tokenizer, prompt_texts, max_seq_len)
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_width = input_ids.shape[1]
    results: list[dict[str, Any]] = []
    for row, record in enumerate(records):
        new_tokens = generated[row, prompt_width:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = parse_option_letter(raw_output)
        results.append(
            {
                "record_id": record["record_id"],
                "base_id": record["base_id"],
                "split": record["split"],
                "task_family": record["task_family"],
                "transform": record["transform"],
                "condition": condition,
                "raw_output": raw_output,
                "parsed_letter": parsed,
                "parse_failure": parsed is None,
            }
        )
    return results
