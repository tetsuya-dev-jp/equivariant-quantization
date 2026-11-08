import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eq_quant.evaluate.scoring import (
    encode_prompts,
    parse_option_letter,
    score_encoded_batch,
    score_records,
    verify_option_token_ids,
)


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        if text in ("A", "B", "C", "D"):
            return [ord(text)]
        if len(text) == 2 and text[0] == " " and text[1] in "ABCD":
            return [ord(text[1])]
        return [ord(character) for character in text]

    def __call__(
        self,
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    ):
        sequences = [self.encode(text) for text in texts]
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = []
        attention_mask = []
        for sequence in sequences:
            pad_length = max_length - len(sequence)
            input_ids.append([0] * pad_length + sequence)
            attention_mask.append([0] * pad_length + [1] * len(sequence))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ):
        return messages[0]["content"] + "\nassistant\n"


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 8)
        self.position_embedding = nn.Embedding(64, 8)
        self.head = nn.Linear(8, 128)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=0,
    ):
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0)
        hidden = self.embedding(input_ids) + self.position_embedding(position_ids)
        logits = self.head(hidden)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


class FixedLogitsModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))
        self.register_buffer("fixed_logits", logits)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=0,
    ):
        batch_size = input_ids.shape[0]
        return SimpleNamespace(
            logits=self.fixed_logits.to(input_ids.device).expand(batch_size, -1, -1)
        )


def make_record(record_id: str, option_order: list[int]) -> dict:
    correct_canonical = 0
    return {
        "record_id": record_id,
        "base_id": "arithmetic-pilot-0000",
        "split": "pilot",
        "task_family": "arithmetic",
        "transform": "original" if option_order == [0, 1, 2, 3] else "option_permutation",
        "semantic": {
            "structure": {"expression": 7},
            "canonical_options": [7, 1, 2, 3],
        },
        "prompt": f"Compute: 7 ({record_id})",
        "option_order": option_order,
        "correct_option_index": option_order.index(correct_canonical),
    }


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.option_ids = verify_option_token_ids(
            self.tokenizer, sample_context="Compute: 7"
        )

    def test_option_token_verification(self) -> None:
        self.assertEqual(self.option_ids, [65, 66, 67, 68])

        class BrokenTokenizer(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                if text in ("A", "B", "C", "D"):
                    return [1, 2]
                return super().encode(text, add_special_tokens)

        with self.assertRaises(ValueError):
            verify_option_token_ids(BrokenTokenizer())

    def test_left_padding_matches_independent_last_token_oracle(self) -> None:
        sequences = [
            torch.tensor([11, 12, 13, 14, 15], dtype=torch.long),
            torch.tensor([21, 22, 23], dtype=torch.long),
            torch.tensor([31, 32, 33, 34], dtype=torch.long),
        ]
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.zeros((len(sequences), max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, -len(sequence) :] = sequence
            attention_mask[row, -len(sequence) :] = 1

        logits, probabilities, token_counts = score_encoded_batch(
            self.model, input_ids, attention_mask, self.option_ids
        )

        expected_rows = []
        for sequence in sequences:
            positions = torch.arange(len(sequence)).unsqueeze(0)
            oracle = self.model(
                input_ids=sequence.unsqueeze(0),
                attention_mask=torch.ones((1, len(sequence)), dtype=torch.long),
                position_ids=positions,
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :]
            expected_rows.append(oracle[:, self.option_ids])
        expected = torch.cat(expected_rows, dim=0)
        expected_probabilities = torch.softmax(expected, dim=-1)

        torch.testing.assert_close(logits, expected)
        torch.testing.assert_close(probabilities, expected_probabilities)
        self.assertEqual(
            token_counts.tolist(),
            [len(sequence) for sequence in sequences],
        )

    def test_bf16_near_tie_keeps_fp32_probabilities_and_logit_prediction(
        self,
    ) -> None:
        logits = torch.zeros((1, 1, 128), dtype=torch.bfloat16)
        logits[0, 0, 65:69] = torch.tensor(
            [0.0, 0.001, 0.0, 0.0], dtype=torch.bfloat16
        )
        model = FixedLogitsModel(logits)
        record = make_record("bf16-near-tie", [0, 1, 2, 3])

        results = score_records(
            model,
            self.tokenizer,
            [record],
            condition="bf16",
            option_token_ids=self.option_ids,
            batch_size=1,
            max_seq_len=128,
        )
        result = results[0]

        self.assertEqual(result["prediction_display"], 1)
        self.assertGreater(len(set(result["option_probabilities"])), 1)
        expected = torch.softmax(
            logits[0, 0, self.option_ids].float(), dim=-1
        ).tolist()
        for actual, target in zip(
            result["option_probabilities"], expected, strict=True
        ):
            self.assertAlmostEqual(actual, target, places=7)

    def test_right_padding_is_rejected(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        with self.assertRaises(ValueError):
            score_encoded_batch(
                self.model, input_ids, attention_mask, self.option_ids
            )

    def test_batching_equals_unpadded_individual_scoring(self) -> None:
        sequences = [
            torch.randint(1, 120, (7,), dtype=torch.long),
            torch.randint(1, 120, (3,), dtype=torch.long),
            torch.randint(1, 120, (5,), dtype=torch.long),
            torch.randint(1, 120, (6,), dtype=torch.long),
        ]
        max_length = max(len(sequence) for sequence in sequences)
        batch_input_ids = torch.zeros(
            (len(sequences), max_length), dtype=torch.long
        )
        batch_attention_mask = torch.zeros_like(batch_input_ids)
        for row, sequence in enumerate(sequences):
            batch_input_ids[row, -len(sequence) :] = sequence
            batch_attention_mask[row, -len(sequence) :] = 1

        batch_logits, batch_probabilities, batch_counts = score_encoded_batch(
            self.model, batch_input_ids, batch_attention_mask, self.option_ids
        )
        individual_logits = []
        individual_probabilities = []
        individual_counts = []
        for sequence in sequences:
            logits, probabilities, counts = score_encoded_batch(
                self.model,
                sequence.unsqueeze(0),
                torch.ones((1, len(sequence)), dtype=torch.long),
                self.option_ids,
            )
            individual_logits.append(logits)
            individual_probabilities.append(probabilities)
            individual_counts.append(counts)

        torch.testing.assert_close(
            batch_logits, torch.cat(individual_logits, dim=0)
        )
        torch.testing.assert_close(
            batch_probabilities, torch.cat(individual_probabilities, dim=0)
        )
        self.assertEqual(
            batch_counts.tolist(),
            torch.cat(individual_counts, dim=0).tolist(),
        )

    def test_score_records_canonicalization_and_batching(self) -> None:
        records = [
            make_record("a", [0, 1, 2, 3]),
            make_record("b", [2, 0, 3, 1]),
            make_record("c", [1, 3, 0, 2]),
        ]
        results_batch = score_records(
            self.model,
            self.tokenizer,
            records,
            condition="fp",
            option_token_ids=self.option_ids,
            batch_size=3,
            max_seq_len=128,
        )
        results_single = score_records(
            self.model,
            self.tokenizer,
            records,
            condition="fp",
            option_token_ids=self.option_ids,
            batch_size=1,
            max_seq_len=128,
        )
        self.assertEqual(results_batch, results_single)

        for record, result in zip(records, results_batch, strict=True):
            self.assertEqual(result["option_order"], record["option_order"])
            self.assertEqual(result["labels"], ["A", "B", "C", "D"])
            self.assertEqual(len(result["option_logits"]), 4)
            for display_index, canonical_index in enumerate(
                record["option_order"]
            ):
                self.assertAlmostEqual(
                    result["canonical_probabilities"][canonical_index],
                    result["option_probabilities"][display_index],
                    places=7,
                )
            if record["record_id"] == "b":
                self.assertEqual(record["option_order"], [2, 0, 3, 1])
            self.assertEqual(
                result["canonical_prediction"],
                record["option_order"][result["prediction_display"]],
            )
            self.assertEqual(result["correct"], result["canonical_prediction"] == 0)
            self.assertEqual(result["token_count"], len(record["prompt"]) + 11)

    def test_encode_prompts_rejects_oversized_input(self) -> None:
        with self.assertRaises(ValueError):
            encode_prompts(self.tokenizer, ["long prompt"], max_seq_len=1)

    def test_parse_option_letter(self) -> None:
        self.assertEqual(parse_option_letter("A"), "A")
        self.assertEqual(parse_option_letter("A."), "A")
        self.assertEqual(parse_option_letter("(B)"), "B")
        self.assertEqual(parse_option_letter("d:"), "D")
        self.assertIsNone(parse_option_letter("A correct answer is B"))
        self.assertIsNone(parse_option_letter("Answer: C"))
        self.assertIsNone(parse_option_letter("E"))
        self.assertIsNone(parse_option_letter("no letter"))


if __name__ == "__main__":
    unittest.main()
