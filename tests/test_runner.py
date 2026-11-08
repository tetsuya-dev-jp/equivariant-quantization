import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from eq_quant.evaluate.runner import run_evaluation


class FakeTokenizer:
    padding_side = "right"
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ):
        return messages[0]["content"] + "\nassistant\n"

    def encode(self, text, add_special_tokens=False):
        if text in ("A", "B", "C", "D"):
            return [ord(text)]
        if len(text) == 2 and text[0] == " " and text[1] in "ABCD":
            return [ord(text[1])]
        return [ord(character) for character in text]


def valid_record() -> dict:
    return {
        "record_id": "arithmetic-pilot-0000__original",
        "base_id": "arithmetic-pilot-0000",
        "split": "pilot",
        "task_family": "arithmetic",
        "transform": "original",
        "semantic": {
            "structure": {"expression": 7},
            "canonical_options": [7, 1, 2, 3],
        },
        "prompt": "Compute: 7",
        "option_order": [0, 1, 2, 3],
        "correct_option_index": 0,
    }


def write_records(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(valid_record()) + "\n")


def minimal_config() -> dict:
    return json.loads(
        Path("configs/pilot.json").read_text(encoding="utf-8")
    )


class RunnerTests(unittest.TestCase):
    def test_refuses_accidental_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            records_path = temp_path / "records.jsonl"
            write_records(records_path)
            output_path = temp_path / "output"
            output_path.mkdir()
            (output_path / "run.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                run_evaluation(
                    minimal_config(),
                    records_path,
                    output_path,
                    overwrite=False,
                )

    def test_failed_run_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            records_path = temp_path / "records.jsonl"
            write_records(records_path)
            output_path = temp_path / "output"

            fake_model = nn.Linear(2, 2)
            fake_model.config = SimpleNamespace(_commit_hash=None)
            with mock.patch(
                "eq_quant.evaluate.runner.load_model",
                return_value=(fake_model, FakeTokenizer(), torch.device("cpu")),
            ), mock.patch(
                "eq_quant.evaluate.runner.score_records",
                side_effect=RuntimeError("boom"),
            ), mock.patch(
                "torch.cuda.is_available",
                return_value=False,
            ):
                with self.assertRaises(RuntimeError):
                    run_evaluation(
                        minimal_config(),
                        records_path,
                        output_path,
                        overwrite=False,
                    )

            manifest = json.loads(
                (output_path / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["state"], "failed")
            self.assertEqual(manifest["completed_conditions"], [])
            self.assertIn("RuntimeError: boom", manifest["error"])


if __name__ == "__main__":
    unittest.main()
