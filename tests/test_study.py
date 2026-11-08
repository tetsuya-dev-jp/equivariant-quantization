import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eq_quant import study
from eq_quant.config import load_config


def calibration_config() -> dict:
    return load_config("configs/calibration.json")


def frozen_selection(config: dict) -> dict:
    return {
        "format": study.SELECTION_FORMAT,
        "state": "frozen",
        "model": config["model"],
        "config": config,
        "objectives": {
            objective: {
                "selected_lambda": 1.0,
                "selected": {"lambda": 1.0},
                "selection": {},
                "candidates": [],
            }
            for objective in study.REGULARIZED_OBJECTIVES
        },
    }


def write_completed_fit(
    root: Path,
    config: dict,
    objective: str,
    lambda_value: float | None,
    seed: int,
    state: str = "completed",
) -> Path:
    fit_dir = study._fit_dir(root, objective, lambda_value, seed)
    fit_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "state": state,
        "objective": objective,
        "lambda": lambda_value,
        "fit_seed": seed,
        "config": config,
        "model": config["model"],
        "optimization_elapsed_seconds": 1.0,
    }
    (fit_dir / study.FIT_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (fit_dir / study.SCALES_FILENAME).write_text("scales", encoding="utf-8")
    return fit_dir


ARITHMETIC_SEMANTIC = {
    "structure": {"expression": {"op": "+", "left": 2, "right": 3}},
    "canonical_options": [5, 1, 2, 3],
}


def final_record(
    record_id: str,
    base_id: str,
    transform: str,
    option_order: list[int],
    split: str = "final",
) -> dict:
    return {
        "record_id": record_id,
        "base_id": base_id,
        "split": split,
        "task_family": "arithmetic",
        "transform": transform,
        "semantic": ARITHMETIC_SEMANTIC,
        "prompt": f"prompt {record_id}",
        "option_order": list(option_order),
        "correct_option_index": list(option_order).index(0),
        "rename": None,
        "order": None,
    }


def write_records(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def prediction_row(
    record: dict,
    condition: str,
    prediction_display: int,
) -> dict:
    option_order = list(record["option_order"])
    canonical_prediction = option_order[prediction_display]
    return {
        "record_id": record["record_id"],
        "base_id": record["base_id"],
        "split": record["split"],
        "task_family": record["task_family"],
        "transform": record["transform"],
        "condition": condition,
        "semantic": record["semantic"],
        "option_order": option_order,
        "labels": ["A", "B", "C", "D"],
        "option_logits": [0.0, 0.0, 0.0, 0.0],
        "option_probabilities": [0.25, 0.25, 0.25, 0.25],
        "prediction_display": prediction_display,
        "canonical_prediction": canonical_prediction,
        "correct_option_index": record["correct_option_index"],
        "correct": canonical_prediction == 0,
        "token_count": 1,
    }


class DevelopPhaseTests(unittest.TestCase):
    def test_develop_fits_all_candidates_and_never_evaluates_final(self) -> None:
        config = calibration_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_calls: list[list[str]] = []
            fit_calls: list[tuple[str, float | None, int]] = []
            evaluation_calls: list[str] = []

            def fake_dataset(config, output_dir, split_names, provenance, phase):
                dataset_calls.append(list(split_names))
                return output_dir / "records.jsonl"

            def fake_fit(
                config,
                records_path,
                teacher_cache_path,
                output_dir,
                objective,
                lambda_value,
                seed,
                phase,
            ):
                fit_calls.append((objective, lambda_value, seed))
                return {"state": "completed"}

            def fake_evaluation(
                config,
                records_path,
                scales_path,
                output_dir,
                objective,
                lambda_value,
                seed,
                condition,
                provenance,
                phase,
            ):
                evaluation_calls.append(condition)
                return {"state": "completed", "condition": condition}

            def fake_metrics(config, evaluation_dir, condition):
                if condition.startswith("augmented_kd"):
                    return {
                        "overall_accuracy": 0.80,
                        "held_out_accuracy": 0.70,
                        "held_out_jointly_correct_rate": 0.65,
                        "held_out_consistency": 0.70,
                        "n_bases": 1,
                    }
                if condition.startswith("original_kd"):
                    return {
                        "overall_accuracy": 0.70,
                        "held_out_accuracy": 0.60,
                        "held_out_jointly_correct_rate": 0.55,
                        "held_out_consistency": 0.60,
                        "n_bases": 1,
                    }
                if "lam0.1" in condition:
                    overall, joint = 0.78, 0.60
                elif "lam10" in condition:
                    overall, joint = 0.75, 0.80
                else:
                    overall, joint = 0.80, 0.70
                return {
                    "overall_accuracy": overall,
                    "held_out_accuracy": 0.72,
                    "held_out_jointly_correct_rate": joint,
                    "held_out_consistency": 0.68,
                    "n_bases": 1,
                }

            with mock.patch.object(
                study, "_ensure_dataset", side_effect=fake_dataset
            ), mock.patch.object(
                study, "_ensure_teacher_cache", return_value=root / "cache.pt"
            ), mock.patch.object(
                study, "_ensure_fit", side_effect=fake_fit
            ), mock.patch.object(
                study, "_ensure_calibrated_evaluation", side_effect=fake_evaluation
            ), mock.patch.object(
                study, "_development_metrics", side_effect=fake_metrics
            ), mock.patch.object(
                study,
                "run_evaluation",
                side_effect=AssertionError("develop must not evaluate final data"),
            ) as final_evaluation, mock.patch.object(
                study,
                "_ensure_constant_display_a_control",
                side_effect=AssertionError(
                    "develop must not invoke the final untrained control"
                ),
            ) as constant_control:
                result = study.run_develop(
                    config,
                    root,
                    {"git_revision": "test", "dirty_paths": []},
                )

            final_evaluation.assert_not_called()
            constant_control.assert_not_called()
            self.assertEqual(
                dataset_calls, [["calibration"], ["development"]]
            )
            self.assertEqual(len(fit_calls), 11)
            self.assertEqual(len(evaluation_calls), 11)
            self.assertEqual(
                {objective for objective, _, _ in fit_calls},
                {
                    "original_kd",
                    "augmented_kd",
                    "paired",
                    "random_pair",
                    "kernel",
                },
            )
            self.assertEqual(
                {seed for _, _, seed in fit_calls}, {study.DEVELOPMENT_SEED}
            )

            selection_path = Path(result["selection"])
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            self.assertEqual(selection["state"], "frozen")
            for objective in study.REGULARIZED_OBJECTIVES:
                block = selection["objectives"][objective]
                self.assertEqual(block["selected_lambda"], 1.0)
                self.assertEqual(len(block["candidates"]), 3)


class FinalPhaseTests(unittest.TestCase):
    def test_final_refuses_missing_selection(self) -> None:
        config = calibration_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                study,
                "run_evaluation",
                side_effect=AssertionError("final evaluation must not start"),
            ) as final_evaluation:
                with self.assertRaises(FileNotFoundError):
                    study.run_final(
                        config,
                        root,
                        {"git_revision": "test", "dirty_paths": []},
                    )
            final_evaluation.assert_not_called()

    def test_final_refuses_incomplete_selected_fits(self) -> None:
        config = calibration_config()
        selection = frozen_selection(config)
        seeds = [int(value) for value in config["calibration"]["fit_seeds"]]
        entries = study._selected_entries(selection)
        entries_and_seeds = [
            (entry, seed) for entry in entries for seed in seeds
        ]
        self.assertEqual(len(entries_and_seeds), 15)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            for index, (entry, seed) in enumerate(entries_and_seeds):
                state = "running" if index == len(entries_and_seeds) - 1 else "completed"
                write_completed_fit(
                    root,
                    config,
                    entry["objective"],
                    entry["lambda"],
                    seed,
                    state=state,
                )

            with mock.patch.object(
                study, "_ensure_dataset"
            ) as dataset, mock.patch.object(
                study,
                "run_evaluation",
                side_effect=AssertionError("final evaluation must not start"),
            ) as final_evaluation:
                with self.assertRaises(RuntimeError):
                    study.run_final(
                        config,
                        root,
                        {"git_revision": "test", "dirty_paths": []},
                    )
            dataset.assert_not_called()
            final_evaluation.assert_not_called()


class ConstantDisplayAControlTests(unittest.TestCase):
    def test_constant_display_a_uses_non_self_inverse_option_order(self) -> None:
        permutation = [2, 0, 3, 1]
        twice = [permutation[permutation[index]] for index in range(4)]
        self.assertNotEqual(twice, [0, 1, 2, 3])

        records = [
            final_record("r-original", "b1", "original", [0, 1, 2, 3]),
            final_record("r-permuted", "b1", "entity_rename", permutation),
        ]
        rows = study._constant_display_a_rows(records)
        by_id = {row["record_id"]: row for row in rows}

        original = by_id["r-original"]
        self.assertEqual(original["prediction_display"], 0)
        self.assertEqual(original["canonical_prediction"], 0)
        self.assertTrue(original["correct"])

        permuted = by_id["r-permuted"]
        self.assertEqual(permuted["prediction_display"], 0)
        self.assertEqual(permuted["canonical_prediction"], 2)
        self.assertFalse(permuted["correct"])
        self.assertEqual(permuted["semantic"], ARITHMETIC_SEMANTIC)
        self.assertEqual(permuted["option_order"], permutation)

        for row in rows:
            self.assertEqual(
                row["condition"], study.CONSTANT_DISPLAY_A_CONDITION
            )
            self.assertNotIn("option_logits", row)
            self.assertNotIn("option_probabilities", row)
            self.assertNotIn("canonical_probabilities", row)

    def test_control_refuses_to_overwrite_unmarked_directory(self) -> None:
        config = calibration_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_path = write_records(
                root / "records.jsonl",
                [final_record("r1", "b1", "original", [0, 1, 2, 3])],
            )
            control_dir = root / "control"
            control_dir.mkdir()
            predictions_path = control_dir / study.PREDICTIONS_FILENAME
            predictions_path.write_text("sentinel", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                study._ensure_constant_display_a_control(
                    config,
                    records_path,
                    control_dir,
                    {"git_revision": "test", "dirty_paths": []},
                    "final",
                )
            self.assertEqual(
                predictions_path.read_text(encoding="utf-8"), "sentinel"
            )

    def test_final_writes_constant_display_a_after_frozen_fits(self) -> None:
        config = calibration_config()
        selection = frozen_selection(config)
        seeds = [int(value) for value in config["calibration"]["fit_seeds"]]
        entries_and_seeds = [
            (entry, seed)
            for entry in study._selected_entries(selection)
            for seed in seeds
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            for entry, seed in entries_and_seeds:
                write_completed_fit(
                    root,
                    config,
                    entry["objective"],
                    entry["lambda"],
                    seed,
                )

            records = [
                final_record("r1", "b1", "original", [0, 1, 2, 3]),
                final_record("r2", "b1", "entity_rename", [2, 0, 3, 1]),
            ]
            records_path = write_records(root / "final_records.jsonl", records)

            with mock.patch.object(
                study, "_ensure_dataset", return_value=records_path
            ), mock.patch.object(
                study, "_ensure_baseline_evaluation", return_value={}
            ), mock.patch.object(
                study, "_ensure_calibrated_evaluation", return_value={}
            ), mock.patch.object(
                study,
                "run_evaluation",
                side_effect=AssertionError("final must not load a model"),
            ) as model_evaluation:
                result = study.run_final(
                    config,
                    root,
                    {"git_revision": "test", "dirty_paths": []},
                )
            model_evaluation.assert_not_called()

            control_dir = study._constant_display_a_dir(root)
            self.assertEqual(result["constant_display_a"], str(control_dir))
            predictions_path = control_dir / study.PREDICTIONS_FILENAME
            rows = [
                json.loads(line)
                for line in predictions_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(rows), 2)
            by_id = {row["record_id"]: row for row in rows}
            self.assertEqual(by_id["r2"]["prediction_display"], 0)
            self.assertEqual(by_id["r2"]["canonical_prediction"], 2)
            self.assertFalse(by_id["r2"]["correct"])
            self.assertEqual(
                by_id["r2"]["semantic"], ARITHMETIC_SEMANTIC
            )
            for row in rows:
                self.assertEqual(
                    row["condition"], study.CONSTANT_DISPLAY_A_CONDITION
                )
                self.assertNotIn("option_logits", row)
                self.assertNotIn("option_probabilities", row)

            manifest = json.loads(
                (control_dir / study.UNTRAINED_RULE_MANIFEST).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["format"], study.UNTRAINED_RULE_FORMAT)
            self.assertTrue(manifest["untrained_rule"])
            self.assertFalse(manifest["uses_model"])
            self.assertFalse(manifest["uses_fit"])
            self.assertFalse(manifest["uses_teacher"])
            self.assertFalse(manifest["replaces_paired_vs_augmented"])
            self.assertEqual(manifest["record_count"], 2)
            self.assertEqual(str(manifest["records_path"]), str(records_path))

            before = predictions_path.read_text(encoding="utf-8")
            with mock.patch.object(
                study, "_ensure_dataset", return_value=records_path
            ), mock.patch.object(
                study, "_ensure_baseline_evaluation", return_value={}
            ), mock.patch.object(
                study, "_ensure_calibrated_evaluation", return_value={}
            ):
                study.run_final(
                    config,
                    root,
                    {"git_revision": "test", "dirty_paths": []},
                )
            self.assertEqual(
                predictions_path.read_text(encoding="utf-8"), before
            )


class ReplicateGuardTests(unittest.TestCase):
    def test_replicate_never_invokes_final_data_or_control(self) -> None:
        config = calibration_config()
        selection = frozen_selection(config)
        seeds = [int(value) for value in config["calibration"]["fit_seeds"]]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            calibration_dir = root / "data" / "calibration"
            calibration_dir.mkdir(parents=True)
            for name in ("records.jsonl", "manifest.json", "study_data.json"):
                (calibration_dir / name).write_text("{}", encoding="utf-8")
            cache_dir = root / "teacher-cache"
            cache_dir.mkdir()
            for name in (
                study.CACHE_FILENAME,
                study.CACHE_MANIFEST,
                "study_cache.json",
            ):
                (cache_dir / name).write_text("{}", encoding="utf-8")

            for entry in study._selected_entries(selection):
                write_completed_fit(
                    root,
                    config,
                    entry["objective"],
                    entry["lambda"],
                    study.DEVELOPMENT_SEED,
                )

            with mock.patch.object(
                study,
                "_ensure_dataset",
                side_effect=AssertionError("replicate must not load final data"),
            ) as dataset, mock.patch.object(
                study,
                "_ensure_constant_display_a_control",
                side_effect=AssertionError(
                    "replicate must not invoke the final untrained control"
                ),
            ) as constant_control, mock.patch.object(
                study, "_ensure_fit", return_value={"state": "completed"}
            ):
                result = study.run_replicate(
                    config,
                    root,
                    {"git_revision": "test", "dirty_paths": []},
                )

            dataset.assert_not_called()
            constant_control.assert_not_called()
            self.assertEqual(len(result["fits"]), 10)
            self.assertEqual(len(seeds), 3)


class ReportConstantControlTests(unittest.TestCase):
    def test_report_includes_constant_display_a_summaries(self) -> None:
        config = calibration_config()
        selection = frozen_selection(config)
        seeds = [int(value) for value in config["calibration"]["fit_seeds"]]
        provenance = {"git_revision": "test", "dirty_paths": []}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            for entry in study._selected_entries(selection):
                for seed in seeds:
                    write_completed_fit(
                        root,
                        config,
                        entry["objective"],
                        entry["lambda"],
                        seed,
                    )

            records = [
                final_record("r1", "b1", "original", [0, 1, 2, 3]),
                final_record("r2", "b1", "entity_rename", [2, 0, 3, 1]),
            ]
            records_path = write_records(
                root / "data" / "final_structural" / "records.jsonl",
                records,
            )
            write_json = study.write_json
            write_jsonl = study.write_jsonl
            study_data_dir = root / "data" / "final_structural"
            write_json(
                study_data_dir / "manifest.json", {"records": len(records)}
            )
            write_json(
                study_data_dir / "study_data.json",
                {
                    "format": study.METADATA_FORMAT,
                    "config": config,
                    "splits": ["final", "structural_holdout"],
                },
            )

            baseline_dir = root / "results" / "final" / "baseline"
            baseline_dir.mkdir(parents=True)
            write_jsonl(
                baseline_dir / "predictions.jsonl",
                [
                    prediction_row(record, "fp", 0)
                    for record in records
                ],
            )
            write_json(
                baseline_dir / "run.json",
                {
                    "state": "completed",
                    "model": config["model"],
                    "completed_conditions": ["fp"],
                },
            )
            write_json(
                baseline_dir / "study_evaluation.json",
                {
                    "format": study.METADATA_FORMAT,
                    "config": config,
                    "records_path": str(records_path),
                },
            )

            for entry in study._selected_entries(selection):
                for seed in seeds:
                    condition = study._final_condition(
                        entry["objective"], seed
                    )
                    evaluation_dir = study._final_dir(
                        root, entry["objective"], entry["lambda"], seed
                    )
                    evaluation_dir.mkdir(parents=True)
                    write_jsonl(
                        evaluation_dir / "predictions.jsonl",
                        [
                            prediction_row(record, condition, 0)
                            for record in records
                        ],
                    )
                    write_json(
                        evaluation_dir / study.EVALUATION_MANIFEST,
                        {
                            "state": "completed",
                            "condition": condition,
                            "model": config["model"],
                        },
                    )
                    scales_path = study._fit_dir(
                        root,
                        entry["objective"],
                        entry["lambda"],
                        seed,
                    )
                    write_json(
                        evaluation_dir / "study_evaluation.json",
                        {
                            "format": study.METADATA_FORMAT,
                            "config": config,
                            "records_path": str(records_path),
                            "scales_path": str(scales_path),
                            "objective": entry["objective"],
                            "lambda": entry["lambda"],
                            "fit_seed": seed,
                            "condition": condition,
                        },
                    )

            study._ensure_constant_display_a_control(
                config,
                records_path,
                study._constant_display_a_dir(root),
                provenance,
                "final",
            )

            study.run_report(config, root, provenance)

            summaries = json.loads(
                (root / "report" / "summaries.json").read_text(
                    encoding="utf-8"
                )
            )
            control = study.CONSTANT_DISPLAY_A_CONDITION
            self.assertIn(control, summaries["conditions"])
            self.assertIn(control, summaries["summaries"])
            self.assertIn(control, summaries["condition_index"])
            self.assertTrue(
                summaries["condition_index"][control]["untrained_rule"]
            )
            self.assertEqual(
                summaries["untrained_rule_conditions"], [control]
            )
            self.assertIn(control, summaries["descriptive_control_conditions"])
            self.assertIn(control, summaries["per_transform"])
            self.assertIn(
                "entity_rename", summaries["per_transform"][control]
            )
            self.assertEqual(
                summaries["per_transform"][control]["entity_rename"][
                    "n_records"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
