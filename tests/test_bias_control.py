import json
import tempfile
import unittest
from pathlib import Path

import torch

from eq_quant.bias_control import (
    CONDITION,
    FIT_FORMAT,
    apply_offsets,
    build_parser,
    fit_offsets,
    main,
)

SEMANTIC = {
    "structure": {"expression": {"op": "+", "left": 2, "right": 3}},
    "canonical_options": [5, 1, 2, 3],
}
ORIGINAL_ORDER = [2, 0, 3, 1]
PERMUTED_ORDER = [1, 3, 0, 2]
BIAS = {"A": -2.5, "B": 0.5, "C": 1.0, "D": 1.0}


def make_record(
    record_id: str,
    base_id: str,
    split: str,
    transform: str,
    option_order: list[int],
) -> dict:
    return {
        "record_id": record_id,
        "base_id": base_id,
        "split": split,
        "task_family": "arithmetic",
        "transform": transform,
        "semantic": SEMANTIC,
        "prompt": f"prompt {record_id}",
        "option_order": list(option_order),
        "correct_option_index": list(option_order).index(0),
        "rename": None,
        "order": None,
    }


def teacher_logits(record: dict) -> list[float]:
    correct_display = record["correct_option_index"]
    logits = [-1.0, 0.5, 0.0, -0.5]
    logits[correct_display] = 2.0
    return logits


def prediction_row(record: dict, condition: str, logits: list[float]) -> dict:
    return {
        "record_id": record["record_id"],
        "base_id": record["base_id"],
        "split": record["split"],
        "task_family": record["task_family"],
        "transform": record["transform"],
        "condition": condition,
        "semantic": record["semantic"],
        "option_order": list(record["option_order"]),
        "labels": ["A", "B", "C", "D"],
        "option_token_ids": [32, 33, 34, 35],
        "option_logits": list(logits),
        "correct_option_index": record["correct_option_index"],
    }


def shifted_w4(record: dict, row_shift: float) -> list[float]:
    fp_logits = teacher_logits(record)
    return [
        fp_logit - BIAS[label] + row_shift
        for label, fp_logit in zip("ABCD", fp_logits, strict=True)
    ]


def fit_rows() -> tuple[list[dict], list[dict]]:
    fp_rows = []
    w4_rows = []
    records = [
        make_record("b1-original", "b1", "calibration", "original", ORIGINAL_ORDER),
        make_record(
            "b1-option", "b1", "calibration", "option_permutation", PERMUTED_ORDER
        ),
        make_record("b2-original", "b2", "calibration", "original", PERMUTED_ORDER),
        make_record(
            "b2-option", "b2", "calibration", "option_permutation", ORIGINAL_ORDER
        ),
    ]
    for index, record in enumerate(records):
        fp_rows.append(prediction_row(record, "fp", teacher_logits(record)))
        w4_rows.append(
            prediction_row(
                record,
                "w4",
                shifted_w4(record, row_shift=0.7 + 0.3 * index),
            )
        )
    return fp_rows, w4_rows


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def write_bundle(directory: Path, w4_rows: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for condition in ("fp", "w4", "w8", "w3"):
        write_jsonl(directory / f"predictions_{condition}.jsonl", w4_rows)
    return directory


def empty_fit_payload(offsets: dict[str, float]) -> dict:
    return {
        "format": FIT_FORMAT,
        "method": "test",
        "offsets": offsets,
        "fit": {"split": "calibration", "transforms": ["original"], "record_count": 1},
    }


class FitTests(unittest.TestCase):
    def test_fit_recovers_known_zero_sum_label_offsets(self) -> None:
        fp_rows, w4_rows = fit_rows()

        payload = fit_offsets(fp_rows, w4_rows)

        self.assertEqual(payload["fit"]["record_count"], 4)
        self.assertEqual(payload["fit"]["base_count"], 2)
        for label in "ABCD":
            self.assertAlmostEqual(payload["offsets"][label], BIAS[label], places=9)
        self.assertAlmostEqual(sum(payload["offsets"].values()), 0.0, places=12)
        self.assertFalse(payload["targets"]["uses_oracle_labels"])
        self.assertFalse(payload["targets"]["uses_prediction_correctness"])

    def test_apply_recovers_teacher_relative_logits_and_prediction(self) -> None:
        fp_rows, w4_rows = fit_rows()
        payload = fit_offsets(fp_rows, w4_rows)

        record = make_record(
            "eval-original", "b1", "final", "original", ORIGINAL_ORDER
        )
        raw_w4 = shifted_w4(record, row_shift=-1.3)
        row = prediction_row(record, "w4", raw_w4)

        result = apply_offsets([row], payload)[0]

        self.assertEqual(result["condition"], CONDITION)
        self.assertEqual(result["raw_option_logits"], raw_w4)
        expected = torch.tensor(teacher_logits(record), dtype=torch.float32)
        adjusted = torch.tensor(result["option_logits"], dtype=torch.float32)
        expected_centered = expected - expected.mean()
        adjusted_centered = adjusted - adjusted.mean()
        self.assertTrue(torch.allclose(adjusted_centered, expected_centered, atol=1e-6))
        self.assertEqual(result["prediction_display"], record["correct_option_index"])
        self.assertEqual(result["canonical_prediction"], 0)
        self.assertTrue(result["correct"])
        self.assertEqual(result["source_condition"], "w4")

    def test_fit_selects_only_original_and_option_permutation(self) -> None:
        fp_rows, w4_rows = fit_rows()
        extra_record = make_record(
            "b1-rename", "b1", "calibration", "entity_rename", ORIGINAL_ORDER
        )
        fp_rows.append(prediction_row(extra_record, "fp", [9.0, 9.0, 9.0, 9.0]))
        w4_rows.append(prediction_row(extra_record, "w4", [-9.0, -9.0, -9.0, -9.0]))

        payload = fit_offsets(fp_rows, w4_rows)

        self.assertEqual(payload["fit"]["record_count"], 4)
        self.assertNotIn("entity_rename", payload["fit"]["by_transform"])
        self.assertAlmostEqual(payload["offsets"]["A"], BIAS["A"], places=9)

    def test_fit_rejects_mixed_splits_instead_of_filtering(self) -> None:
        fp_rows, w4_rows = fit_rows()
        w4_rows[0] = dict(w4_rows[0], split="development")

        with self.assertRaisesRegex(ValueError, "accepts only 'calibration'"):
            fit_offsets(fp_rows, w4_rows)

    def test_fit_rejects_duplicate_record_ids(self) -> None:
        fp_rows, w4_rows = fit_rows()
        fp_rows.append(dict(fp_rows[0]))

        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            fit_offsets(fp_rows, w4_rows)

    def test_fit_rejects_unequal_support(self) -> None:
        fp_rows, w4_rows = fit_rows()
        w4_rows.pop()

        with self.assertRaisesRegex(ValueError, "support differs"):
            fit_offsets(fp_rows, w4_rows)

    def test_fit_rejects_mismatched_metadata(self) -> None:
        fp_rows, w4_rows = fit_rows()
        w4_rows[0] = dict(w4_rows[0], base_id="different-base")

        with self.assertRaisesRegex(ValueError, "metadata differs"):
            fit_offsets(fp_rows, w4_rows)

    def test_fit_refuses_to_overwrite_existing_output(self) -> None:
        fp_rows, w4_rows = fit_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fp_path = write_jsonl(root / "fp.jsonl", fp_rows)
            w4_path = write_jsonl(root / "w4.jsonl", w4_rows)
            output = root / "offsets.json"
            output.write_text("sentinel", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                main(
                    [
                        "fit",
                        "--fp",
                        str(fp_path),
                        "--w4",
                        str(w4_path),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel")


class ApplyTests(unittest.TestCase):
    def test_non_self_inverse_option_mapping_is_canonicalized(self) -> None:
        record = make_record(
            "eval-order", "b1", "final", "original", ORIGINAL_ORDER
        )
        row = prediction_row(record, "w4", [0.0, 1.0, 3.0, 2.0])
        offsets = {label: 0.0 for label in "ABCD"}

        result = apply_offsets([row], empty_fit_payload(offsets))[0]

        self.assertEqual(result["prediction_display"], 2)
        self.assertEqual(result["canonical_prediction"], 3)
        self.assertFalse(result["correct"])
        display = torch.tensor(result["option_probabilities"], dtype=torch.float32)
        canonical = result["canonical_probabilities"]
        for display_index, canonical_index in enumerate(ORIGINAL_ORDER):
            self.assertAlmostEqual(
                canonical[canonical_index], display[display_index].item(), places=12
            )

    def test_apply_refuses_to_overwrite_existing_output(self) -> None:
        record = make_record(
            "eval-order", "b1", "final", "original", ORIGINAL_ORDER
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = write_bundle(
                root / "bundle", [prediction_row(record, "w4", [0, 1, 3, 2])]
            )
            offsets_path = root / "offsets.json"
            offsets_path.write_text(
                json.dumps(empty_fit_payload({label: 0.0 for label in "ABCD"})),
                encoding="utf-8",
            )
            output = root / "predictions.jsonl"
            output.write_text("sentinel", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                main(
                    [
                        "apply",
                        "--offsets",
                        str(offsets_path),
                        "--bundle",
                        str(bundle),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel")


class CliTests(unittest.TestCase):
    def test_fit_then_apply_from_bundle(self) -> None:
        fp_rows, w4_rows = fit_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fp_path = write_jsonl(root / "fp.jsonl", fp_rows)
            w4_path = write_jsonl(root / "w4.jsonl", w4_rows)
            offsets_path = root / "offsets.json"

            main(
                [
                    "fit",
                    "--fp",
                    str(fp_path),
                    "--w4",
                    str(w4_path),
                    "--output",
                    str(offsets_path),
                ]
            )

            record = make_record(
                "eval-original", "b1", "pilot", "original", ORIGINAL_ORDER
            )
            bundle = write_bundle(
                root / "bundle",
                [prediction_row(record, "w4", shifted_w4(record, 0.0))],
            )
            output = root / "predictions.jsonl"
            main(
                [
                    "apply",
                    "--offsets",
                    str(offsets_path),
                    "--bundle",
                    str(bundle),
                    "--output",
                    str(output),
                ]
            )

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["condition"], CONDITION)
            self.assertEqual(rows[0]["split"], "pilot")
            self.assertTrue(rows[0]["correct"])

    def test_module_parser_has_fit_and_apply(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--help"])


if __name__ == "__main__":
    unittest.main()
