import csv
import json
import tempfile
import unittest
from pathlib import Path

from eq_quant.analyze.stats import analyze_predictions


def arithmetic_base() -> dict:
    return {
        "base_id": "arithmetic-pilot-0000",
        "split": "pilot",
        "task_family": "arithmetic",
        "semantic": {
            "structure": {"expression": 7},
            "canonical_options": [7, 1, 2, 3],
        },
        "transforms": ("original", "option_permutation", "entity_rename"),
    }


def boolean_base() -> dict:
    return {
        "base_id": "boolean-pilot-0000",
        "split": "pilot",
        "task_family": "boolean",
        "semantic": {
            "structure": {
                "variables": ["p", "q"],
                "variable_count": 2,
                "expression": {
                    "op": "and",
                    "left": "p",
                    "right": {"op": "not", "child": "q"},
                },
            },
            "canonical_options": [
                [True, False],
                [False, False],
                [False, True],
                [True, True],
            ],
        },
        "transforms": (
            "original",
            "option_permutation",
            "entity_rename",
            "order_permutation",
        ),
    }


def make_prediction(
    base: dict,
    transform: str,
    condition: str,
    canonical_prediction: int,
) -> dict:
    option_order = [1, 0, 2, 3] if transform == "option_permutation" else [0, 1, 2, 3]
    return {
        "record_id": f"{base['base_id']}__{transform}",
        "base_id": base["base_id"],
        "split": base["split"],
        "task_family": base["task_family"],
        "transform": transform,
        "condition": condition,
        "semantic": base["semantic"],
        "option_order": option_order,
        "correct_option_index": option_order.index(0),
        "canonical_prediction": canonical_prediction,
        "correct": canonical_prediction == 0,
    }


def synthetic_predictions() -> list[dict]:
    base = arithmetic_base()
    predictions = []
    outcomes = {
        "fp": (0, 0, 0),
        "w4": (0, 1, 0),
    }
    for condition, values in outcomes.items():
        for transform, value in zip(base["transforms"], values, strict=True):
            predictions.append(
                make_prediction(base, transform, condition, value)
            )
    return predictions


def mixed_family_predictions() -> list[dict]:
    arithmetic = arithmetic_base()
    boolean = boolean_base()
    predictions = []
    for base, fp_values, w4_values in (
        (arithmetic, (0, 0, 0), (0, 1, 2)),
        (boolean, (0, 0, 0, 0), (0, 1, 0, 0)),
    ):
        for condition, values in (("fp", fp_values), ("w4", w4_values)):
            for transform, value in zip(
                base["transforms"], values, strict=True
            ):
                predictions.append(
                    make_prediction(base, transform, condition, value)
                )
    return predictions


def reference_predictions() -> list[dict]:
    base = arithmetic_base()
    predictions = []
    for condition, values in (
        ("augmented_s17", (0, 0, 0)),
        ("paired_s17", (0, 1, 1)),
    ):
        for transform, value in zip(base["transforms"], values, strict=True):
            predictions.append(
                make_prediction(base, transform, condition, value)
            )
    return predictions


def split_composition_predictions() -> list[dict]:
    arithmetic = arithmetic_base()
    pilot = {
        **arithmetic,
        "base_id": "arithmetic-pilot-0000",
        "split": "pilot",
        "transforms": ("original", "option_permutation", "entity_rename"),
    }
    final = {
        **arithmetic,
        "base_id": "arithmetic-final-0000",
        "split": "final",
        "transforms": (
            "original",
            "option_permutation",
            "entity_rename",
            "option_permutation+entity_rename",
        ),
    }
    predictions = []
    for base in (pilot, final):
        for condition in ("fp", "w4"):
            for transform in base["transforms"]:
                predictions.append(
                    make_prediction(base, transform, condition, 0)
                )
    return predictions


def write_predictions(predictions: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in predictions:
            handle.write(json.dumps(entry) + "\n")


class AnalyzeTests(unittest.TestCase):
    def test_paired_summaries_and_bootstrap(self) -> None:
        predictions = synthetic_predictions()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            predictions_path = temp_path / "predictions.jsonl"
            write_predictions(predictions, predictions_path)

            result = analyze_predictions(
                predictions_path,
                temp_path / "analysis",
                n_boot=50,
                seed=7,
            )

            overall_fp = next(
                row
                for row in result["summary"]
                if row["split"] == "all"
                and row["task_family"] == "all"
                and row["condition"] == "fp"
                and row["transform"] == "all"
            )
            self.assertEqual(overall_fp["n_bases"], 1)
            self.assertEqual(overall_fp["n_records"], 3)
            self.assertAlmostEqual(overall_fp["accuracy_macro"], 1.0)
            self.assertAlmostEqual(overall_fp["accuracy_micro"], 1.0)
            self.assertAlmostEqual(overall_fp["consistency_macro"], 1.0)
            self.assertAlmostEqual(overall_fp["consistency_micro"], 1.0)
            self.assertEqual(overall_fp["jointly_correct_pairs"], 2)

            overall_w4 = next(
                row
                for row in result["summary"]
                if row["split"] == "all"
                and row["task_family"] == "all"
                and row["condition"] == "w4"
                and row["transform"] == "all"
            )
            self.assertAlmostEqual(overall_w4["accuracy_macro"], 2 / 3)
            self.assertAlmostEqual(overall_w4["consistency_macro"], 1 / 2)

            comparison = next(
                row
                for row in result["comparisons"]
                if row["split"] == "all"
                and row["task_family"] == "all"
                and row["condition"] == "w4"
                and row["transform"] == "all"
            )
            self.assertEqual(comparison["broken_consistent_pairs"], 1)
            self.assertEqual(comparison["repaired_consistent_pairs"], 0)
            self.assertEqual(comparison["answer_flips"], 1)
            self.assertAlmostEqual(comparison["delta_accuracy_macro"], -1 / 3)
            self.assertAlmostEqual(comparison["delta_consistency_macro"], -1 / 2)
            self.assertEqual(comparison["consistent_but_wrong_pairs"], 0)
            self.assertTrue(comparison["union_bound_satisfied"])

            arithmetic_order = next(
                row
                for row in result["summary"]
                if row["task_family"] == "arithmetic"
                and row["condition"] == "fp"
                and row["transform"] == "order_permutation"
            )
            self.assertFalse(arithmetic_order["available"])
            self.assertEqual(arithmetic_order["n_bases"], 0)
            self.assertIsNone(arithmetic_order["accuracy_macro"])

            self.assertTrue((temp_path / "analysis" / "analysis.json").exists())
            self.assertTrue((temp_path / "analysis" / "summary.csv").exists())
            self.assertTrue((temp_path / "analysis" / "comparisons.csv").exists())
            self.assertTrue((temp_path / "analysis" / "bootstrap.csv").exists())

            with (temp_path / "analysis" / "summary.csv").open(
                "r", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            csv_fp = next(
                row
                for row in csv_rows
                if row["split"] == "all"
                and row["task_family"] == "all"
                and row["condition"] == "fp"
                and row["transform"] == "all"
            )
            self.assertEqual(csv_fp["jointly_correct_pairs"], "2")
            self.assertAlmostEqual(
                float(csv_fp["accuracy_macro"]),
                overall_fp["accuracy_macro"],
            )

            self.assertTrue(result["bootstrap_intervals"])
            for row in result["bootstrap_intervals"]:
                self.assertLessEqual(row["ci_low"], row["ci_high"])
                self.assertEqual(row["n_boot"], 50)

    def test_macro_averaging_by_base_and_transformation_rows(self) -> None:
        predictions = mixed_family_predictions()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            predictions_path = temp_path / "predictions.jsonl"
            write_predictions(predictions, predictions_path)
            result = analyze_predictions(
                predictions_path,
                temp_path / "analysis",
                n_boot=20,
                seed=3,
            )

            comparison = next(
                row
                for row in result["comparisons"]
                if row["split"] == "all"
                and row["task_family"] == "all"
                and row["condition"] == "w4"
                and row["transform"] == "all"
            )
            self.assertEqual(comparison["broken_consistent_pairs"], 3)
            self.assertAlmostEqual(
                comparison["broken_consistent_pair_rate_macro"], 2 / 3
            )
            self.assertAlmostEqual(
                comparison["delta_accuracy_macro"], -11 / 24
            )

            option_permutation = next(
                row
                for row in result["comparisons"]
                if row["split"] == "all"
                and row["task_family"] == "all"
                and row["condition"] == "w4"
                and row["transform"] == "option_permutation"
            )
            self.assertEqual(option_permutation["broken_consistent_pairs"], 2)
            self.assertAlmostEqual(
                option_permutation["broken_consistent_pair_rate_macro"], 1.0
            )

    def test_consistent_but_wrong_pairs(self) -> None:
        base = arithmetic_base()
        predictions = []
        for condition, values in (
            ("fp", (0, 0, 0)),
            ("w4", (1, 1, 0)),
        ):
            for transform, value in zip(
                base["transforms"], values, strict=True
            ):
                predictions.append(
                    make_prediction(base, transform, condition, value)
                )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            predictions_path = temp_path / "predictions.jsonl"
            write_predictions(predictions, predictions_path)
            result = analyze_predictions(
                predictions_path,
                temp_path / "analysis",
                n_boot=10,
                seed=5,
            )
            comparison = next(
                row
                for row in result["comparisons"]
                if row["split"] == "all"
                and row["task_family"] == "all"
                and row["condition"] == "w4"
                and row["transform"] == "all"
            )
            self.assertEqual(comparison["broken_consistent_pairs"], 1)
            self.assertEqual(comparison["consistent_but_wrong_pairs"], 1)

    def test_availability_honors_split_and_family_support(self) -> None:
        predictions = split_composition_predictions()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            predictions_path = temp_path / "predictions.jsonl"
            write_predictions(predictions, predictions_path)
            result = analyze_predictions(
                predictions_path,
                temp_path / "analysis",
                n_boot=5,
                seed=2,
            )

            summary = {
                (
                    row["split"],
                    row["task_family"],
                    row["condition"],
                    row["transform"],
                ): row
                for row in result["summary"]
            }
            comparisons = {
                (
                    row["split"],
                    row["task_family"],
                    row["condition"],
                    row["transform"],
                ): row
                for row in result["comparisons"]
            }

            pilot_composition = summary[
                (
                    "pilot",
                    "arithmetic",
                    "fp",
                    "option_permutation+entity_rename",
                )
            ]
            self.assertFalse(pilot_composition["available"])
            self.assertEqual(pilot_composition["n_bases"], 0)
            self.assertIsNone(pilot_composition["accuracy_macro"])
            self.assertFalse(
                comparisons[
                    (
                        "pilot",
                        "arithmetic",
                        "w4",
                        "option_permutation+entity_rename",
                    )
                ]["available"]
            )

            final_composition = summary[
                (
                    "final",
                    "arithmetic",
                    "fp",
                    "option_permutation+entity_rename",
                )
            ]
            self.assertTrue(final_composition["available"])
            self.assertEqual(final_composition["n_bases"], 1)
            self.assertAlmostEqual(final_composition["accuracy_macro"], 1.0)
            self.assertTrue(
                comparisons[
                    (
                        "final",
                        "arithmetic",
                        "w4",
                        "option_permutation+entity_rename",
                    )
                ]["available"]
            )

            final_triple = summary[
                (
                    "final",
                    "arithmetic",
                    "fp",
                    "option_permutation+entity_rename+order_permutation",
                )
            ]
            self.assertFalse(final_triple["available"])
            self.assertEqual(final_triple["n_bases"], 0)

            pooled_composition = summary[
                (
                    "all",
                    "arithmetic",
                    "fp",
                    "option_permutation+entity_rename",
                )
            ]
            self.assertTrue(pooled_composition["available"])
            self.assertEqual(pooled_composition["n_bases"], 1)

    def test_custom_reference_comparison_and_bootstrap(self) -> None:
        predictions = reference_predictions()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            predictions_path = temp_path / "predictions.jsonl"
            write_predictions(predictions, predictions_path)

            result = analyze_predictions(
                predictions_path,
                temp_path / "analysis",
                n_boot=20,
                seed=13,
                conditions=["augmented_s17", "paired_s17"],
                reference="augmented_s17",
            )

            comparison = next(
                row
                for row in result["comparisons"]
                if row["split"] == "all"
                and row["task_family"] == "all"
                and row["condition"] == "paired_s17"
                and row["transform"] == "all"
            )
            self.assertEqual(comparison["reference"], "augmented_s17")
            self.assertAlmostEqual(comparison["delta_accuracy_macro"], -2 / 3)
            self.assertAlmostEqual(comparison["delta_consistency_macro"], -1.0)
            self.assertEqual(comparison["broken_consistent_pairs"], 2)

            bootstrap = next(
                row
                for row in result["bootstrap_intervals"]
                if row["split"] == "all"
                and row["task_family"] == "all"
                and row["condition"] == "paired_s17"
                and row["transform"] == "all"
                and row["metric"] == "delta_accuracy_macro"
            )
            self.assertEqual(bootstrap["reference"], "augmented_s17")
            self.assertEqual(bootstrap["n_boot"], 20)
            self.assertAlmostEqual(bootstrap["estimate"], -2 / 3)
            self.assertAlmostEqual(bootstrap["ci_low"], -2 / 3)
            self.assertAlmostEqual(bootstrap["ci_high"], -2 / 3)

    def test_bootstrap_is_deterministic(self) -> None:
        predictions = synthetic_predictions()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            predictions_path = temp_path / "predictions.jsonl"
            write_predictions(predictions, predictions_path)

            first = analyze_predictions(
                predictions_path, temp_path / "a", n_boot=20, seed=11
            )
            second = analyze_predictions(
                predictions_path, temp_path / "b", n_boot=20, seed=11
            )
            self.assertEqual(
                first["bootstrap_intervals"], second["bootstrap_intervals"]
            )


if __name__ == "__main__":
    unittest.main()
