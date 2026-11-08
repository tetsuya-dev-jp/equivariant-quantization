import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from eq_quant.calibrate.basis import (
    build_rtn_basis,
    export_scales,
    load_scales,
    replace_linear_modules,
)
from eq_quant.calibrate.objectives import (
    canonical_option_distributions,
    cosine_gram_mse,
    full_vocabulary_kl,
    jensen_shannon,
    paired_js,
    random_pair_js,
)
from eq_quant.calibrate.records import (
    record_metadata,
    select_calibration_pairs,
    validate_calibration_records,
)
from eq_quant.calibrate.teacher import (
    _teacher_batch,
    cache_teacher,
    load_teacher_cache,
)
from eq_quant.calibrate.train import (
    activate_calibration_mode,
    run_calibration,
)
from eq_quant.evaluate.quantize import quantize_weight_rtn
from eq_quant.evaluate.scoring import build_chat_prompt, encode_prompts, score_records


class TinyTokenizer:
    padding_side = "right"
    pad_token_id = 0

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


class TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(128, 8)
        self.layer = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 128)
        self.config = SimpleNamespace(_commit_hash="tiny-revision")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=0,
        output_hidden_states=False,
    ):
        hidden = self.embed(input_ids)
        hidden = self.layer(hidden)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        if output_hidden_states:
            return SimpleNamespace(logits=logits, hidden_states=(hidden,))
        return SimpleNamespace(logits=logits)


def arithmetic_record(record_id: str, base_id: str, option_order: list[int]):
    correct_canonical = 0
    return {
        "record_id": record_id,
        "base_id": base_id,
        "split": "calibration",
        "task_family": "arithmetic",
        "transform": (
            "original" if option_order == [0, 1, 2, 3] else "option_permutation"
        ),
        "semantic": {
            "structure": {"expression": 7},
            "canonical_options": [7, 1, 2, 3],
        },
        "prompt": f"Compute: 7 ({record_id})",
        "option_order": option_order,
        "correct_option_index": option_order.index(correct_canonical),
    }


def calibration_records() -> list[dict]:
    return [
        arithmetic_record("base0__original", "base0", [0, 1, 2, 3]),
        arithmetic_record("base0__permutation", "base0", [2, 0, 3, 1]),
        arithmetic_record("base1__original", "base1", [0, 1, 2, 3]),
        arithmetic_record("base1__permutation", "base1", [3, 1, 0, 2]),
    ]


def tiny_config() -> dict:
    config = json.loads(
        Path("configs/calibration.json").read_text(encoding="utf-8")
    )
    config["model"]["id"] = "tiny/model"
    config["model"]["revision"] = "tiny-revision"
    config["scoring"]["batch_size"] = 2
    config["scoring"]["max_seq_len"] = 128
    config["calibration"].update(
        {
            "bases_per_batch": 2,
            "prompts_per_batch": 4,
            "total_updates": 1,
            "gradient_checkpointing": False,
        }
    )
    return config


def write_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_teacher_cache(
    path: Path,
    records: list[dict],
    model: TinyCausalModel,
    tokenizer: TinyTokenizer,
) -> None:
    texts = [build_chat_prompt(tokenizer, record["prompt"]) for record in records]
    encoded = encode_prompts(tokenizer, texts, max_seq_len=128)
    logits, hidden = _teacher_batch(
        model, encoded["input_ids"], encoded["attention_mask"]
    )
    torch.save(
        {
            "format": "eq_quant_teacher_cache_v1",
            "model": {
                "id": "tiny/model",
                "revision": "tiny-revision",
                "dtype": "bfloat16",
            },
            "scoring": {
                "max_seq_len": 128,
                "option_labels": ["A", "B", "C", "D"],
            },
            "records": [record_metadata(record) for record in records],
            "record_ids": [record["record_id"] for record in records],
            "token_counts": encoded["attention_mask"].sum(dim=1).to(torch.long),
            "logits": logits,
            "hidden": hidden,
        },
        path,
    )


class CalibratedLinearTests(unittest.TestCase):
    def test_reconstruction_matches_rtn_with_zero_and_tail_groups(self) -> None:
        torch.manual_seed(0)
        weight = torch.tensor(
            [[0.0, 0.0, 1.0, -2.0, 3.0]], dtype=torch.float32
        )
        module = nn.Linear(5, 1, bias=True)
        with torch.no_grad():
            module.weight.copy_(weight)
            module.bias.copy_(torch.tensor([0.25]))

        class Holder(nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.layer = linear

        holder = Holder(module)
        basis = build_rtn_basis(holder, bits=4, group_size=2)
        modules = replace_linear_modules(holder, basis)
        calibrated = modules["layer"]

        rtn = quantize_weight_rtn(weight, bits=4, group_size=2)
        expected = rtn.dequantized.to(torch.bfloat16).to(torch.float32)
        self.assertTrue(
            torch.equal(calibrated.reconstructed_weight(torch.float32), expected)
        )
        self.assertTrue(torch.equal(calibrated.log_scale_deltas, torch.zeros_like(calibrated.log_scale_deltas)))
        self.assertTrue(
            torch.equal(calibrated.rtn_scales, rtn.scales.to(torch.float32))
        )

        with torch.no_grad():
            calibrated.log_scale_deltas.add_(
                torch.tensor([[-0.5, 0.2, 0.4]], dtype=torch.float32)
            )
        reconstructed = calibrated.reconstructed_weight(torch.float32)
        self.assertTrue(torch.equal(reconstructed[:, :2], torch.zeros(1, 2)))
        self.assertFalse(torch.equal(reconstructed[:, 2:], expected[:, 2:]))

        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
        output = calibrated(inputs)
        manual = torch.nn.functional.linear(
            inputs, reconstructed, module.bias.detach()
        )
        self.assertTrue(torch.equal(output, manual))

    def test_initialization_matches_rtn_over_bits_zero_and_tail_groups(self) -> None:
        torch.manual_seed(3)
        for bits in (8, 4, 3):
            for group_size in (2, 4, 8):
                weight = torch.randn(3, 7, dtype=torch.float32) * 0.8
                weight[:, :group_size] = 0.0
                module = nn.Linear(7, 3, bias=False)
                with torch.no_grad():
                    module.weight.copy_(weight)

                class Holder(nn.Module):
                    def __init__(self, linear):
                        super().__init__()
                        self.layer = linear

                holder = Holder(module)
                basis = build_rtn_basis(holder, bits, group_size)
                modules = replace_linear_modules(holder, basis)
                calibrated = modules["layer"]

                rtn = quantize_weight_rtn(weight, bits, group_size)
                expected = rtn.dequantized.to(torch.bfloat16).to(torch.float32)
                self.assertTrue(
                    torch.equal(
                        calibrated.reconstructed_weight(torch.float32),
                        expected,
                    ),
                    msg=f"bits={bits} group_size={group_size}",
                )

                with torch.no_grad():
                    calibrated.log_scale_deltas.add_(
                        torch.randn_like(calibrated.log_scale_deltas) * 0.2
                    )
                exported = export_scales(modules)

                second_module = nn.Linear(7, 3, bias=False)
                with torch.no_grad():
                    second_module.weight.copy_(weight)
                second_holder = Holder(second_module)
                second_basis = build_rtn_basis(second_holder, bits, group_size)
                second_modules = replace_linear_modules(second_holder, second_basis)
                load_scales(second_modules, exported)
                self.assertTrue(
                    torch.equal(
                        calibrated.reconstructed_weight(torch.float32),
                        second_modules["layer"].reconstructed_weight(
                            torch.float32
                        ),
                    )
                )

    def test_exported_scales_reproduce_calibrated_logits_exactly(self) -> None:
        torch.manual_seed(1)
        model = TinyCausalModel()
        original_state = copy.deepcopy(model.state_dict())
        input_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)

        basis = build_rtn_basis(model, bits=4, group_size=128)
        modules = replace_linear_modules(model, basis)
        with torch.no_grad():
            for module in modules.values():
                module.log_scale_deltas.add_(0.37)

        logits_in_memory = model(input_ids, logits_to_keep=1).logits
        exported = export_scales(modules)

        second_model = TinyCausalModel()
        second_model.load_state_dict(original_state)
        second_basis = build_rtn_basis(second_model, bits=4, group_size=128)
        second_modules = replace_linear_modules(second_model, second_basis)
        load_scales(second_modules, exported)
        logits_reloaded = second_model(input_ids, logits_to_keep=1).logits

        self.assertTrue(torch.equal(logits_in_memory, logits_reloaded))

    def test_shared_scorer_accepts_calibrated_model(self) -> None:
        torch.manual_seed(2)
        model = TinyCausalModel()
        basis = build_rtn_basis(model, bits=4, group_size=128)
        replace_linear_modules(model, basis)
        tokenizer = TinyTokenizer()
        tokenizer.padding_side = "left"
        record = arithmetic_record("base0__original", "base0", [0, 1, 2, 3])

        predictions = score_records(
            model=model,
            tokenizer=tokenizer,
            records=[record],
            condition="calibrated-test",
            option_token_ids=[65, 66, 67, 68],
            batch_size=1,
            max_seq_len=128,
        )
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["condition"], "calibrated-test")
        self.assertEqual(len(predictions[0]["canonical_probabilities"]), 4)


class ObjectiveTests(unittest.TestCase):
    def test_full_vocabulary_kl_is_zero_for_identical_predictions(self) -> None:
        logits = torch.tensor([[0.1, -0.3, 2.0, 0.0]], dtype=torch.float32)
        self.assertAlmostEqual(
            float(full_vocabulary_kl(logits, logits.clone())), 0.0, places=7
        )
        other = torch.tensor([[0.0, 2.0, -1.0, 0.5]], dtype=torch.float32)
        self.assertGreater(float(full_vocabulary_kl(logits, other)), 0.0)

    def test_canonicalization_uses_record_option_order(self) -> None:
        student_logits = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        record = arithmetic_record("x", "base", [2, 0, 3, 1])
        option_ids = [4, 5, 6, 7]
        student_logits[0, option_ids] = torch.tensor(
            [0.0, 0.0, 5.0, 1.0], dtype=torch.float32
        )
        canonical = canonical_option_distributions(
            student_logits, [record], option_ids
        )
        # Display index 2 is canonical identity 3, so canonical slot 3 wins.
        self.assertEqual(int(torch.argmax(canonical[0])), 3)

    def test_js_and_random_pair_shift(self) -> None:
        probabilities = torch.tensor(
            [[0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]], dtype=torch.float32
        )
        self.assertAlmostEqual(
            float(paired_js(probabilities, num_bases=1)), 0.0, places=7
        )

        # orig0 == variant1 and orig1 == variant0, so only the random-pair
        # control can be exactly zero while the same-base pairing is positive.
        cross = torch.tensor(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.7],
                [0.1, 0.1, 0.1, 0.7],
                [0.7, 0.1, 0.1, 0.1],
            ],
            dtype=torch.float32,
        )
        self.assertGreater(float(paired_js(cross, num_bases=2)), 0.0)
        self.assertAlmostEqual(
            float(random_pair_js(cross, num_bases=2)), 0.0, places=7
        )

        p = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        q = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        self.assertGreater(float(jensen_shannon(p, q)[0]), 0.0)

    def test_js_degenerate_distributions_have_finite_gradients(self) -> None:
        p = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True
        )
        q = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True
        )
        value = jensen_shannon(p, q)
        self.assertTrue(torch.isfinite(value).all())
        self.assertAlmostEqual(
            float(value.detach()[0]),
            float(torch.log(torch.tensor(2.0))),
            places=6,
        )
        value.sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_kernel_loss_is_zero_for_identical_hidden(self) -> None:
        hidden = torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float32
        )
        self.assertAlmostEqual(
            float(cosine_gram_mse(hidden, hidden.clone())), 0.0, places=7
        )
        other = torch.tensor(
            [[0.0, 1.0], [2.0, 0.0], [-1.0, 1.0]], dtype=torch.float32
        )
        self.assertGreater(float(cosine_gram_mse(hidden, other)), 0.0)


class RecordSelectionTests(unittest.TestCase):
    def test_pairs_and_non_calibration_refusal(self) -> None:
        pairs = select_calibration_pairs(calibration_records())
        self.assertEqual([pair.base_id for pair in pairs], ["base0", "base1"])

        pilot = [dict(calibration_records()[0], split="pilot")]
        with self.assertRaises(ValueError):
            validate_calibration_records(pilot)

    def test_identity_option_permutation_is_rejected(self) -> None:
        records = calibration_records()
        records[1]["option_order"] = [0, 1, 2, 3]
        with self.assertRaises(ValueError):
            select_calibration_pairs(records)

    def test_non_calibration_split_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_calibration_records(calibration_records(), "development")

    def test_changed_prompt_with_same_record_ids_is_rejected(self) -> None:
        records = calibration_records()
        pairs = select_calibration_pairs(records)
        ordered = [record for pair in pairs for record in pair.ordered_records()]
        tokenizer = TinyTokenizer()
        tokenizer.padding_side = "left"
        model = TinyCausalModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "teacher_cache.pt"
            write_teacher_cache(cache_path, ordered, model, tokenizer)

            changed = copy.deepcopy(ordered)
            changed[0]["prompt"] = changed[0]["prompt"] + " changed"
            with self.assertRaises(ValueError):
                load_teacher_cache(
                    cache_path,
                    expected_records=changed,
                    model_config={
                        "id": "tiny/model",
                        "revision": "tiny-revision",
                        "dtype": "bfloat16",
                    },
                    scoring_config={
                        "max_seq_len": 128,
                        "option_labels": ["A", "B", "C", "D"],
                    },
                )

            load_teacher_cache(
                cache_path,
                expected_records=ordered,
                model_config={
                    "id": "tiny/model",
                    "revision": "tiny-revision",
                    "dtype": "bfloat16",
                },
                scoring_config={
                    "max_seq_len": 128,
                    "option_labels": ["A", "B", "C", "D"],
                },
            )


class TeacherCacheTests(unittest.TestCase):
    def _held_out_record(
        self, record_id: str, base_id: str, transform: str
    ) -> dict:
        record = arithmetic_record(record_id, base_id, [0, 1, 2, 3])
        record["transform"] = transform
        record["prompt"] = f"Held-out {transform}: {record_id}"
        return record

    def test_cache_matches_selected_pairs_for_shuffled_mixed_transform_input(
        self,
    ) -> None:
        supplied = calibration_records()
        supplied.append(
            self._held_out_record("base0__rename", "base0", "entity_rename")
        )
        supplied.append(
            self._held_out_record("base1__order", "base1", "order_permutation")
        )
        shuffled = [supplied[index] for index in (4, 1, 5, 0, 3, 2)]

        tokenizer = TinyTokenizer()
        model = TinyCausalModel()
        config = tiny_config()
        config["scoring"]["batch_size"] = 2

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            records_path = temp_path / "records.jsonl"
            write_records(records_path, shuffled)
            output_dir = temp_path / "cache"

            with mock.patch(
                "eq_quant.calibrate.teacher.load_model",
                return_value=(model, tokenizer, torch.device("cpu")),
            ):
                manifest = cache_teacher(
                    config=config,
                    records_path=records_path,
                    output_dir=output_dir,
                )

            pairs = select_calibration_pairs(shuffled)
            ordered = [
                record for pair in pairs for record in pair.ordered_records()
            ]
            expected_metadata = [record_metadata(record) for record in ordered]
            expected_ids = [record["record_id"] for record in ordered]

            payload = torch.load(
                output_dir / "teacher_cache.pt", map_location="cpu"
            )
            self.assertEqual(payload["records"], expected_metadata)
            self.assertEqual(payload["record_ids"], expected_ids)
            self.assertEqual(manifest["records"]["count"], len(ordered))
            self.assertEqual(manifest["records"]["bases"], len(pairs))
            self.assertEqual(payload["logits"].shape[0], len(ordered))
            self.assertEqual(payload["hidden"].shape[0], len(ordered))
            self.assertEqual(payload["token_counts"].shape[0], len(ordered))

            tokenizer.padding_side = "left"
            for index, record in enumerate(ordered):
                text = build_chat_prompt(tokenizer, record["prompt"])
                encoded = encode_prompts(tokenizer, [text], max_seq_len=128)
                row_logits, row_hidden = _teacher_batch(
                    model, encoded["input_ids"], encoded["attention_mask"]
                )
                self.assertTrue(
                    torch.allclose(payload["logits"][index], row_logits[0])
                )
                self.assertTrue(
                    torch.allclose(payload["hidden"][index], row_hidden[0])
                )
                self.assertEqual(
                    int(payload["token_counts"][index]),
                    int(encoded["attention_mask"].sum(dim=1)[0]),
                )

    def test_mixed_split_input_is_rejected(self) -> None:
        records = calibration_records()
        records[1] = dict(records[1], split="final")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            records_path = temp_path / "records.jsonl"
            write_records(records_path, records)

            with mock.patch(
                "eq_quant.calibrate.teacher.load_model",
                side_effect=AssertionError("model must not be loaded"),
            ):
                with self.assertRaises(ValueError):
                    cache_teacher(
                        config=tiny_config(),
                        records_path=records_path,
                        output_dir=temp_path / "cache",
                    )


class CalibrationRunTests(unittest.TestCase):
    def _prepare(self, temp_path: Path):
        records = calibration_records()
        records_path = temp_path / "records.jsonl"
        write_records(records_path, records)

        pairs = select_calibration_pairs(records)
        ordered = [record for pair in pairs for record in pair.ordered_records()]
        tokenizer = TinyTokenizer()
        tokenizer.padding_side = "left"
        teacher_model = TinyCausalModel()
        cache_path = temp_path / "teacher_cache.pt"
        write_teacher_cache(cache_path, ordered, teacher_model, tokenizer)
        return records_path, cache_path, teacher_model

    def test_paired_fit_and_calibrated_evaluation_round_trip(self) -> None:
        from eq_quant.calibrate.evaluate import run_calibrated_evaluation

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            records_path, cache_path, teacher_model = self._prepare(temp_path)
            original_state = copy.deepcopy(teacher_model.state_dict())

            output_path = temp_path / "fit"
            fit_model = TinyCausalModel()
            fit_model.load_state_dict(original_state)
            tokenizer = TinyTokenizer()
            with mock.patch(
                "eq_quant.calibrate.train.load_model",
                return_value=(fit_model, tokenizer, torch.device("cpu")),
            ):
                manifest = run_calibration(
                    config=tiny_config(),
                    records_path=records_path,
                    teacher_cache_path=cache_path,
                    output_dir=output_path,
                    objective="paired",
                    lambda_value=1.0,
                    seed=17,
                )

            self.assertEqual(manifest["state"], "completed")
            self.assertEqual(len(manifest["steps"]), 1)
            self.assertEqual(manifest["lambda"], 1.0)
            self.assertTrue((output_path / "scales.pt").is_file())
            self.assertGreater(manifest["trainable_parameters"], 0)
            self.assertEqual(
                manifest["token_counts"]["nonpadding_total"],
                manifest["steps"][0]["nonpadding_tokens"],
            )
            self.assertGreaterEqual(
                manifest["token_counts"]["padded_total"],
                manifest["token_counts"]["nonpadding_total"],
            )
            self.assertEqual(manifest["config"], tiny_config())
            self.assertEqual(len(manifest["batch_order"]), 1)
            self.assertEqual(
                len(manifest["batch_order"][0]["base_ids"]), 2
            )
            self.assertEqual(
                len(manifest["batch_order"][0]["record_ids"]), 4
            )
            self.assertTrue(torch.isfinite(
                torch.tensor(manifest["steps"][0]["gradient_norm"])
            ))

            with self.assertRaises(FileExistsError):
                with mock.patch(
                    "eq_quant.calibrate.train.load_model",
                    return_value=(fit_model, tokenizer, torch.device("cpu")),
                ):
                    run_calibration(
                        config=tiny_config(),
                        records_path=records_path,
                        teacher_cache_path=cache_path,
                        output_dir=output_path,
                        objective="paired",
                        lambda_value=1.0,
                        seed=17,
                    )

            eval_model = TinyCausalModel()
            eval_model.load_state_dict(original_state)
            eval_output = temp_path / "eval"
            with mock.patch(
                "eq_quant.calibrate.evaluate.load_model",
                return_value=(eval_model, tokenizer, torch.device("cpu")),
            ):
                eval_manifest = run_calibrated_evaluation(
                    config=tiny_config(),
                    records_path=records_path,
                    scales_path=output_path,
                    output_dir=eval_output,
                    condition="paired_w4_s17",
                )
            self.assertEqual(eval_manifest["state"], "completed")
            self.assertEqual(eval_manifest["condition"], "paired_w4_s17")
            self.assertEqual(eval_manifest["predictions"], 4)
            self.assertTrue((eval_output / "predictions.jsonl").is_file())

    def test_all_five_objectives_run_one_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            records_path, cache_path, teacher_model = self._prepare(temp_path)
            original_state = copy.deepcopy(teacher_model.state_dict())
            tokenizer = TinyTokenizer()
            for objective in (
                "original_kd",
                "augmented_kd",
                "paired",
                "random_pair",
                "kernel",
            ):
                fit_model = TinyCausalModel()
                fit_model.load_state_dict(original_state)
                lambda_value = (
                    None if objective.endswith("kd") else 0.5
                )
                with mock.patch(
                    "eq_quant.calibrate.train.load_model",
                    return_value=(fit_model, tokenizer, torch.device("cpu")),
                ):
                    manifest = run_calibration(
                        config=tiny_config(),
                        records_path=records_path,
                        teacher_cache_path=cache_path,
                        output_dir=temp_path / objective,
                        objective=objective,
                        lambda_value=lambda_value,
                        seed=17,
                    )
                self.assertEqual(manifest["state"], "completed")
                self.assertEqual(manifest["objective"], objective)
                self.assertEqual(len(manifest["steps"]), 1)
                self.assertGreater(
                    manifest["steps"][0]["nonpadding_tokens"], 0
                )
                if lambda_value is None:
                    self.assertEqual(
                        manifest["steps"][0]["regularizer_loss"], 0.0
                    )
                else:
                    self.assertGreaterEqual(
                        manifest["steps"][0]["regularizer_loss"], 0.0
                    )

    def test_regularized_objective_requires_explicit_lambda(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            records = calibration_records()
            records_path = temp_path / "records.jsonl"
            write_records(records_path, records)
            with self.assertRaises(ValueError):
                run_calibration(
                    config=tiny_config(),
                    records_path=records_path,
                    teacher_cache_path=temp_path / "missing.pt",
                    output_dir=temp_path / "fit",
                    objective="paired",
                    lambda_value=None,
                    seed=17,
                )

    def test_calibration_refuses_non_calibration_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            records = [dict(record, split="pilot") for record in calibration_records()]
            records_path = temp_path / "records.jsonl"
            write_records(records_path, records)
            with self.assertRaises(ValueError):
                run_calibration(
                    config=tiny_config(),
                    records_path=records_path,
                    teacher_cache_path=temp_path / "missing.pt",
                    output_dir=temp_path / "fit",
                    objective="augmented_kd",
                    lambda_value=None,
                    seed=17,
                )

    def test_configured_split_other_than_calibration_is_refused(self) -> None:
        config = tiny_config()
        config["calibration"]["split"] = "development"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with self.assertRaises(ValueError):
                run_calibration(
                    config=config,
                    records_path=temp_path / "records.jsonl",
                    teacher_cache_path=temp_path / "missing.pt",
                    output_dir=temp_path / "fit",
                    objective="augmented_kd",
                    lambda_value=None,
                    seed=17,
                )


class NativeTransformersTests(unittest.TestCase):
    """Exercise real Transformers layers without downloads or a GPU."""

    @staticmethod
    def _tiny_qwen2():
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.1,
        )
        return Qwen2ForCausalLM(config).to(torch.float32)

    @staticmethod
    def _tiny_llama():
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.1,
        )
        return LlamaForCausalLM(config).to(torch.float32)

    def _assert_export_round_trip(self, model) -> None:
        state = copy.deepcopy(model.state_dict())
        basis = build_rtn_basis(model, bits=4, group_size=8)
        modules = replace_linear_modules(model, basis)

        first_module = sorted(modules)[0]
        rtn = quantize_weight_rtn(state[first_module + ".weight"], bits=4, group_size=8)
        expected = rtn.dequantized.to(torch.bfloat16).to(torch.float32)
        self.assertTrue(
            torch.equal(
                modules[first_module].reconstructed_weight(torch.float32),
                expected,
            )
        )

        with torch.no_grad():
            for index, name in enumerate(sorted(modules)):
                modules[name].log_scale_deltas.add_(0.03 * (index + 1))

        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            original_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=1,
            ).logits
        exported = export_scales(modules)

        reloaded = type(model)(model.config).to(torch.float32)
        reloaded.load_state_dict(state)
        reloaded.eval()
        reloaded_basis = build_rtn_basis(reloaded, bits=4, group_size=8)
        reloaded_modules = replace_linear_modules(reloaded, reloaded_basis)
        load_scales(reloaded_modules, exported)
        with torch.no_grad():
            reloaded_logits = reloaded(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=1,
            ).logits
        self.assertTrue(torch.equal(original_logits, reloaded_logits))

    def test_qwen2_exported_predictions_round_trip(self) -> None:
        self._assert_export_round_trip(self._tiny_qwen2())

    def test_llama_exported_predictions_round_trip(self) -> None:
        self._assert_export_round_trip(self._tiny_llama())

    def test_llama_activation_zeroes_attention_dropout(self) -> None:
        model = self._tiny_llama()
        activate_calibration_mode(model, gradient_checkpointing=False)
        self.assertTrue(model.training)
        self.assertEqual(model.config.attention_dropout, 0.0)
        for module in model.modules():
            if hasattr(module, "attention_dropout"):
                self.assertEqual(module.attention_dropout, 0.0)

    def test_qwen2_checkpointing_reaches_early_and_late_scale_gradients(self) -> None:
        model = self._tiny_qwen2()
        activate_calibration_mode(model, gradient_checkpointing=True)

        self.assertTrue(model.training)
        self.assertEqual(model.config.attention_dropout, 0.0)
        for module in model.modules():
            if hasattr(module, "attention_dropout"):
                self.assertEqual(module.attention_dropout, 0.0)
        self.assertTrue(
            all(layer.gradient_checkpointing for layer in model.model.layers)
        )

        checkpoint_calls = []
        for layer in model.model.layers:
            original = layer._gradient_checkpointing_func

            def counted(func, *args, _original=original, **kwargs):
                checkpoint_calls.append(1)
                return _original(func, *args, **kwargs)

            layer._gradient_checkpointing_func = counted

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        basis = build_rtn_basis(model, bits=4, group_size=8)
        modules = replace_linear_modules(model, basis)

        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits
        logits.float().pow(2).mean().backward()

        self.assertEqual(len(checkpoint_calls), len(model.model.layers))
        for layer_index in (0, 1):
            layer_modules = [
                module
                for name, module in modules.items()
                if name.startswith(f"model.layers.{layer_index}.")
            ]
            self.assertTrue(layer_modules)
            for module in layer_modules:
                gradient = module.log_scale_deltas.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue(
                any(
                    module.log_scale_deltas.grad.abs().sum() > 0
                    for module in layer_modules
                )
            )


if __name__ == "__main__":
    unittest.main()
