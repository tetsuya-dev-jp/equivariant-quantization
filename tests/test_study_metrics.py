import unittest

from eq_quant.data.oracles import canonical_answer
from eq_quant.study_metrics import (
    compare_conditions,
    group_by_condition_and_base,
    recompute_correct,
    select_lambda,
    summarize_condition,
    summarize_conditions,
)


def make_semantic(options=None, expression=7):
    if options is None:
        options = [7, 1, 2, 3]
    return {
        "structure": {"expression": expression},
        "canonical_options": list(options),
    }


def make_row(
    base_id,
    condition,
    transform,
    prediction,
    *,
    semantic=None,
    split="development",
    task_family="arithmetic",
    option_order=None,
):
    semantic = semantic or make_semantic()
    options = semantic["canonical_options"]
    order = (
        list(option_order)
        if option_order is not None
        else list(range(len(options)))
    )
    gold = canonical_answer(task_family, semantic["structure"], options)
    return {
        "record_id": f"{base_id}__{condition}__{transform}",
        "base_id": base_id,
        "split": split,
        "task_family": task_family,
        "transform": transform,
        "condition": condition,
        "semantic": semantic,
        "option_order": order,
        "canonical_prediction": prediction,
        "prediction_display": order.index(prediction),
        "correct_option_index": order.index(gold),
        "correct": prediction == gold,
    }


def base_with_rows(base_id, condition, outcomes, **kwargs):
    return [
        make_row(base_id, condition, transform, prediction, **kwargs)
        for transform, prediction in outcomes
    ]


class GroupingAndSummaryTests(unittest.TestCase):
    def test_unequal_transforms_use_base_macro_and_endpoint_absence(self):
        predictions = [
            *base_with_rows(
                "b1",
                "w4",
                (
                    ("original", 0),
                    ("entity_rename", 0),
                    ("order_permutation", 1),
                    ("option_permutation+entity_rename", 0),
                ),
            ),
            *base_with_rows(
                "b2",
                "w4",
                (
                    ("original", 0),
                    ("entity_rename", 1),
                ),
            ),
        ]

        summary = summarize_condition(predictions, "w4")

        self.assertEqual(summary["n_bases"], 2)
        self.assertEqual(summary["n_records"], 6)
        self.assertEqual(summary["n_correct"], 4)
        self.assertAlmostEqual(summary["accuracy_macro"], 0.625)
        self.assertAlmostEqual(summary["accuracy_micro"], 4 / 6)
        self.assertAlmostEqual(summary["original_accuracy_macro"], 1.0)

        all_variants = summary["endpoints"]["all_variants"]
        self.assertEqual(all_variants["n_bases"], 2)
        self.assertEqual(all_variants["n_pairs"], 4)
        self.assertAlmostEqual(all_variants["accuracy_macro"], 1 / 3)
        self.assertAlmostEqual(all_variants["consistency_macro"], 1 / 3)

        held_out = summary["endpoints"]["held_out_singles"]
        self.assertEqual(held_out["n_bases"], 2)
        self.assertEqual(held_out["n_pairs"], 3)
        self.assertAlmostEqual(held_out["accuracy_macro"], 1 / 4)
        self.assertAlmostEqual(held_out["consistency_macro"], 1 / 4)
        self.assertAlmostEqual(
            held_out["jointly_correct_rate_macro"], 1 / 4
        )
        self.assertAlmostEqual(
            held_out["consistent_wrong_rate_macro"], 0.0
        )

        compositions = summary["endpoints"]["compositions"]
        self.assertEqual(compositions["n_bases"], 1)
        self.assertEqual(compositions["n_pairs"], 1)
        self.assertAlmostEqual(compositions["accuracy_macro"], 1.0)
        self.assertAlmostEqual(compositions["consistency_macro"], 1.0)

        self.assertIsNone(summary["bases"]["b2"]["endpoints"]["compositions"])
        self.assertIsNotNone(
            summary["bases"]["b1"]["endpoints"]["compositions"]
        )

    def test_consistent_wrong_pairs_are_distinct_from_joint_correctness(self):
        predictions = base_with_rows(
            "b1",
            "w4",
            (
                ("original", 2),
                ("entity_rename", 2),
                ("order_permutation", 1),
            ),
        )

        summary = summarize_condition(predictions, "w4")
        all_variants = summary["endpoints"]["all_variants"]

        self.assertEqual(all_variants["n_pairs"], 2)
        self.assertEqual(all_variants["n_consistent"], 1)
        self.assertEqual(all_variants["n_jointly_correct"], 0)
        self.assertEqual(all_variants["n_consistent_wrong"], 1)
        self.assertAlmostEqual(all_variants["consistency_macro"], 0.5)
        self.assertAlmostEqual(
            all_variants["jointly_correct_rate_macro"], 0.0
        )
        self.assertAlmostEqual(
            all_variants["consistent_wrong_rate_macro"], 0.5
        )

    def test_non_self_inverse_option_mapping_is_recomputed_from_gold(self):
        semantic = make_semantic(options=[1, 7, 2, 3])
        order = [2, 0, 3, 1]
        correct_row = make_row(
            "b1",
            "w4",
            "original",
            1,
            semantic=semantic,
            option_order=order,
        )
        self.assertTrue(recompute_correct(correct_row))
        self.assertEqual(correct_row["prediction_display"], 3)
        self.assertEqual(correct_row["correct_option_index"], 3)

        wrong_row = make_row(
            "b1",
            "w4",
            "entity_rename",
            0,
            semantic=semantic,
            option_order=order,
        )
        self.assertFalse(recompute_correct(wrong_row))
        self.assertEqual(wrong_row["prediction_display"], 1)

        inconsistent = dict(correct_row)
        inconsistent["canonical_prediction"] = 0
        with self.assertRaises(ValueError):
            recompute_correct(inconsistent)

        mislabelled = dict(correct_row)
        mislabelled["correct"] = False
        with self.assertRaises(ValueError):
            recompute_correct(mislabelled)

        bad_index = dict(correct_row)
        bad_index["correct_option_index"] = 0
        with self.assertRaises(ValueError):
            recompute_correct(bad_index)

    def test_grouping_rejects_duplicate_transform_missing_original_and_metadata(self):
        semantic = make_semantic()
        valid = make_row("b1", "w4", "original", 0, semantic=semantic)

        with self.assertRaises(ValueError):
            group_by_condition_and_base([valid, dict(valid)])

        with self.assertRaises(ValueError):
            group_by_condition_and_base(
                [make_row("b1", "w4", "entity_rename", 0)]
            )

        other_split = make_row(
            "b1",
            "w3",
            "original",
            0,
            semantic=semantic,
            split="final",
        )
        with self.assertRaises(ValueError):
            group_by_condition_and_base([valid, other_split])

        other_semantic = make_semantic(options=[7, 1, 2, 3], expression=8)
        with self.assertRaises(ValueError):
            group_by_condition_and_base(
                [valid, make_row("b1", "w3", "original", 0, semantic=other_semantic)]
            )

    def test_summarize_conditions_requires_present_condition(self):
        predictions = base_with_rows("b1", "w4", (("original", 0),))
        with self.assertRaises(ValueError):
            summarize_conditions(predictions, ["w3"])


class PairedComparisonTests(unittest.TestCase):
    def comparison_predictions(self):
        predictions = []
        predictions.extend(
            base_with_rows(
                "b1",
                "augmented",
                (
                    ("original", 0),
                    ("entity_rename", 0),
                ),
            )
        )
        predictions.extend(
            base_with_rows(
                "b1",
                "paired",
                (
                    ("original", 0),
                    ("entity_rename", 1),
                ),
            )
        )
        predictions.extend(
            base_with_rows(
                "b1",
                "fp",
                (
                    ("original", 0),
                    ("entity_rename", 0),
                ),
            )
        )
        predictions.extend(
            base_with_rows(
                "b2",
                "augmented",
                (
                    ("original", 1),
                    ("entity_rename", 1),
                ),
            )
        )
        predictions.extend(
            base_with_rows(
                "b2",
                "paired",
                (
                    ("original", 0),
                    ("entity_rename", 0),
                ),
            )
        )
        predictions.extend(
            base_with_rows(
                "b2",
                "fp",
                (
                    ("original", 0),
                    ("entity_rename", 1),
                ),
            )
        )
        return predictions

    def test_known_paired_deltas_and_matching_support(self):
        result = compare_conditions(
            self.comparison_predictions(),
            condition="paired",
            baseline="augmented",
            reference="fp",
            n_boot=40,
            seed=4108,
        )

        self.assertEqual(result["direction"], "condition_minus_baseline")
        self.assertEqual(result["n_bases_compared"], 2)
        self.assertEqual(result["n_bases_only_condition"], 0)
        self.assertEqual(result["n_bases_only_baseline"], 0)
        self.assertEqual(result["support"]["b1"], ["entity_rename", "original"])

        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["accuracy"]["delta"], 0.25)
        self.assertAlmostEqual(
            metrics["original_accuracy"]["delta"], 0.5
        )
        self.assertAlmostEqual(
            metrics["transformed_accuracy"]["delta"], 0.0
        )
        self.assertAlmostEqual(metrics["consistency"]["delta"], -0.5)
        self.assertAlmostEqual(
            metrics["jointly_correct_rate"]["delta"], 0.0
        )
        self.assertAlmostEqual(
            metrics["consistent_wrong_rate"]["delta"], -0.5
        )
        self.assertAlmostEqual(
            metrics["held_out_accuracy"]["delta"], 0.0
        )
        self.assertAlmostEqual(
            metrics["held_out_consistency"]["delta"], -0.5
        )
        self.assertAlmostEqual(
            metrics["held_out_jointly_correct_rate"]["delta"], 0.0
        )
        self.assertAlmostEqual(
            metrics["held_out_consistent_wrong_rate"]["delta"], -0.5
        )
        self.assertAlmostEqual(
            metrics["teacher_fidelity"]["delta"], -0.25
        )
        self.assertAlmostEqual(
            metrics["held_out_teacher_fidelity"]["delta"], -1.0
        )
        self.assertIsNone(metrics["composition_accuracy"]["delta"])
        self.assertEqual(metrics["composition_accuracy"]["n_bases"], 0)
        self.assertIsNone(
            metrics["composition_teacher_fidelity"]["delta"]
        )

        for metric in metrics.values():
            if metric["ci_low"] is not None:
                self.assertLessEqual(metric["ci_low"], metric["ci_high"])
                self.assertEqual(metric["n_bases"], 2)

    def test_comparison_requires_reference_to_add_fidelity(self):
        result = compare_conditions(
            self.comparison_predictions(),
            condition="paired",
            baseline="augmented",
            n_boot=10,
            seed=1,
        )
        self.assertNotIn("teacher_fidelity", result["metrics"])

    def test_bootstrap_is_deterministic(self):
        first = compare_conditions(
            self.comparison_predictions(),
            condition="paired",
            baseline="augmented",
            reference="fp",
            n_boot=100,
            seed=99,
        )
        second = compare_conditions(
            self.comparison_predictions(),
            condition="paired",
            baseline="augmented",
            reference="fp",
            n_boot=100,
            seed=99,
        )
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["bootstrap"], second["bootstrap"])

    def test_comparison_rejects_missing_conditions_and_no_shared_bases(self):
        predictions = self.comparison_predictions()
        with self.assertRaises(ValueError):
            compare_conditions(predictions, "missing", "augmented", n_boot=5)
        with self.assertRaises(ValueError):
            compare_conditions(
                predictions, "paired", "augmented", reference="missing", n_boot=5
            )

        disjoint = base_with_rows("b9", "other", (("original", 0),))
        with self.assertRaises(ValueError):
            compare_conditions(
                [*predictions, *disjoint],
                condition="other",
                baseline="paired",
                n_boot=5,
            )

    def test_comparison_rejects_unequal_base_sets(self):
        predictions = [
            *self.comparison_predictions(),
            *base_with_rows(
                "b3",
                "paired",
                (
                    ("original", 0),
                    ("entity_rename", 0),
                ),
            ),
        ]
        with self.assertRaisesRegex(ValueError, "identical base support"):
            compare_conditions(
                predictions,
                condition="paired",
                baseline="augmented",
                n_boot=10,
            )

    def test_comparison_rejects_unequal_transform_sets(self):
        predictions = [
            *self.comparison_predictions(),
            make_row("b1", "paired", "order_permutation", 0),
        ]
        with self.assertRaisesRegex(
            ValueError, "identical transform support"
        ):
            compare_conditions(
                predictions,
                condition="paired",
                baseline="augmented",
                n_boot=10,
            )

    def test_comparison_rejects_incomplete_reference_support(self):
        predictions = [
            row
            for row in self.comparison_predictions()
            if not (
                row["condition"] == "fp"
                and row["base_id"] == "b2"
                and row["transform"] == "entity_rename"
            )
        ]
        with self.assertRaisesRegex(ValueError, "incomplete reference support"):
            compare_conditions(
                predictions,
                condition="paired",
                baseline="augmented",
                reference="fp",
                n_boot=10,
            )


class LambdaSelectionTests(unittest.TestCase):
    def test_constrained_selection_uses_joint_then_consistency_then_lambda(self):
        candidates = [
            {
                "lambda": 0.1,
                "overall_accuracy": 0.80,
                "held_out_jointly_correct_rate": 0.60,
                "held_out_consistency": 0.90,
            },
            {
                "lambda": 1.0,
                "overall_accuracy": 0.85,
                "held_out_jointly_correct_rate": 0.62,
                "held_out_consistency": 0.68,
            },
            {
                "lambda": 10.0,
                "overall_accuracy": 0.79,
                "held_out_jointly_correct_rate": 0.99,
                "held_out_consistency": 0.99,
            },
        ]

        result = select_lambda(candidates, augmented_overall_accuracy=0.81)

        self.assertEqual(result["mode"], "accuracy_constraint")
        self.assertAlmostEqual(result["accuracy_threshold"], 0.80)
        self.assertAlmostEqual(result["selected_lambda"], 1.0)
        self.assertEqual(len(result["candidates"]), 3)
        self.assertAlmostEqual(candidates[0]["overall_accuracy"], 0.80)

    def test_constrained_boundary_and_smaller_lambda_tie_break(self):
        candidates = [
            {
                "lambda": 1.0,
                "overall_accuracy": 0.80,
                "held_out_jointly_correct_rate": 0.5,
                "held_out_consistency": 0.6,
            },
            {
                "lambda": 0.1,
                "overall_accuracy": 0.80,
                "held_out_jointly_correct_rate": 0.5,
                "held_out_consistency": 0.6,
            },
        ]
        result = select_lambda(candidates, augmented_overall_accuracy=0.81)
        self.assertEqual(result["mode"], "accuracy_constraint")
        self.assertAlmostEqual(result["selected_lambda"], 0.1)

    def test_fallback_uses_accuracy_then_joint_then_consistency_then_lambda(self):
        candidates = [
            {
                "lambda": 0.1,
                "overall_accuracy": 0.80,
                "held_out_jointly_correct_rate": 0.5,
                "held_out_consistency": 0.9,
            },
            {
                "lambda": 1.0,
                "overall_accuracy": 0.80,
                "held_out_jointly_correct_rate": 0.6,
                "held_out_consistency": 0.5,
            },
            {
                "lambda": 10.0,
                "overall_accuracy": 0.80,
                "held_out_jointly_correct_rate": 0.6,
                "held_out_consistency": 0.6,
            },
            {
                "lambda": 100.0,
                "overall_accuracy": 0.70,
                "held_out_jointly_correct_rate": 0.99,
                "held_out_consistency": 0.99,
            },
        ]
        result = select_lambda(candidates, augmented_overall_accuracy=0.90)
        self.assertEqual(result["mode"], "fallback")
        self.assertAlmostEqual(result["selected_lambda"], 10.0)

    def test_selection_rejects_missing_or_duplicate_candidates(self):
        with self.assertRaises(ValueError):
            select_lambda([], augmented_overall_accuracy=0.8)
        with self.assertRaises(ValueError):
            select_lambda(
                [
                    {
                        "lambda": 1.0,
                        "overall_accuracy": 0.8,
                        "held_out_jointly_correct_rate": None,
                        "held_out_consistency": 0.7,
                    }
                ],
                augmented_overall_accuracy=0.8,
            )
        duplicate = {
            "lambda": 1.0,
            "overall_accuracy": 0.8,
            "held_out_jointly_correct_rate": 0.6,
            "held_out_consistency": 0.7,
        }
        with self.assertRaises(ValueError):
            select_lambda(
                [duplicate, dict(duplicate)], augmented_overall_accuracy=0.8
            )


if __name__ == "__main__":
    unittest.main()
