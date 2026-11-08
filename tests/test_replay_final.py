import copy
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "replay_final.py"
)
SPEC = importlib.util.spec_from_file_location("replay_final", MODULE_PATH)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)

SEMANTIC = {
    "structure": {"expression": {"op": "+", "left": 2, "right": 3}},
    "canonical_options": [5, 1, 2, 3],
}
CONDITION_INDEX = {
    "augmented_kd_s17": {
        "objective": "augmented_kd",
        "lambda": None,
        "fit_seed": 17,
    },
    "fp": {"objective": "baseline", "lambda": None, "fit_seed": None},
    "paired_s17": {"objective": "paired", "lambda": 0.5, "fit_seed": 17},
}


def prediction_row(
    base_id: str,
    transform: str,
    condition: str,
    canonical_prediction: int,
) -> dict:
    return {
        "record_id": f"{base_id}-{condition}-{transform}",
        "base_id": base_id,
        "split": "final",
        "task_family": "arithmetic",
        "transform": transform,
        "condition": condition,
        "semantic": SEMANTIC,
        "option_order": [0, 1, 2, 3],
        "prediction_display": canonical_prediction,
        "canonical_prediction": canonical_prediction,
        "correct_option_index": 0,
        "correct": canonical_prediction == 0,
    }


def synthetic_predictions() -> list[dict]:
    rows: list[dict] = []
    for base_id, paired_entity_prediction in (
        ("base-1", 1),
        ("base-2", 0),
    ):
        for condition in ("fp", "paired_s17", "augmented_kd_s17"):
            rows.append(
                prediction_row(base_id, "original", condition, 0)
            )
            entity_prediction = (
                paired_entity_prediction
                if condition == "paired_s17"
                else 0
            )
            rows.append(
                prediction_row(
                    base_id, "entity_rename", condition, entity_prediction
                )
            )
    return rows


def composition_predictions() -> list[dict]:
    rows: list[dict] = []
    for base_id, paired_composition_prediction in (
        ("composition-1", 1),
        ("composition-2", 0),
    ):
        for condition in ("fp", "paired_s17", "augmented_kd_s17"):
            rows.append(prediction_row(base_id, "original", condition, 0))
            composition_prediction = (
                paired_composition_prediction
                if condition == "paired_s17"
                else 0
            )
            rows.append(
                prediction_row(
                    base_id,
                    "option_permutation+entity_rename",
                    condition,
                    composition_prediction,
                )
            )
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_source(
    source: Path,
    predictions: list[dict],
    summaries: dict,
    comparisons: dict,
    selection: dict | None = None,
) -> None:
    source.mkdir(parents=True, exist_ok=True)
    with gzip.open(source / "predictions.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(source / "summaries.json", summaries)
    write_json(source / "comparisons.json", comparisons)
    if selection is not None:
        write_json(source / "selection.json", selection)


def find_comparison(comparisons: list[dict], split: str, family: str) -> dict:
    for entry in comparisons:
        if entry["split"] == split and entry["task_family"] == family:
            return entry
    raise AssertionError(f"missing comparison {split}/{family}")


class ReplayFinalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = synthetic_predictions()
        self.recomputed = replay.recompute(
            self.predictions,
            condition_index=CONDITION_INDEX,
            bootstrap_samples=40,
            bootstrap_seed=7,
            model_id="synthetic/model",
        )
        self.released_summaries = {
            "format": "eq_quant_study_report_v1",
            "state": "completed",
            "model": {"id": "synthetic/model"},
            "conditions": self.recomputed["conditions"],
            "condition_index": copy.deepcopy(CONDITION_INDEX),
            "summaries": self.recomputed["summaries"],
            "grouped_summaries": self.recomputed["grouped_summaries"],
            "per_transform": self.recomputed["per_transform"],
            "descriptive_control_conditions": self.recomputed[
                "descriptive_control_conditions"
            ],
            "untrained_rule_conditions": self.recomputed[
                "untrained_rule_conditions"
            ],
        }
        self.released_comparisons = {
            "format": "eq_quant_study_report_v1",
            "state": "completed",
            "model": {"id": "synthetic/model"},
            "primary": "paired_vs_augmented_kd_same_seed",
            "evidence": "held_out_singles",
            "reference_condition": "fp",
            "bootstrap": {"unit": "base_id", "samples": 40, "seed": 7},
            "comparisons": self.recomputed["comparisons"],
            "optimization_time": {"note": "not replayed"},
        }
        self.selection = {
            "format": "eq_quant_study_selection_v1",
            "state": "frozen",
            "fit_seeds": [17],
            "config": {
                "model": {"id": "synthetic/model"},
                "analysis": {"bootstrap_samples": 40, "seed": 7},
            },
        }

    def test_recomputes_known_rates_and_paired_delta(self) -> None:
        paired = self.recomputed["summaries"]["paired_s17"]
        self.assertEqual(paired["n_bases"], 2)
        self.assertEqual(paired["n_records"], 4)
        self.assertEqual(paired["n_correct"], 3)
        self.assertAlmostEqual(paired["accuracy_macro"], 0.75, places=12)
        self.assertAlmostEqual(paired["accuracy_micro"], 0.75, places=12)

        comparison = find_comparison(
            self.recomputed["comparisons"], split="all", family="all"
        )
        metrics = comparison["comparison"]["metrics"]
        self.assertAlmostEqual(metrics["accuracy"]["delta"], -0.25, places=12)
        self.assertAlmostEqual(
            metrics["held_out_accuracy"]["delta"], -0.5, places=12
        )
        self.assertAlmostEqual(
            metrics["held_out_consistency"]["delta"], -0.5, places=12
        )
        self.assertEqual(metrics["held_out_accuracy"]["n_bases"], 2)

    def test_replay_passes_on_matching_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            write_source(
                source,
                self.predictions,
                self.released_summaries,
                self.released_comparisons,
                self.selection,
            )
            verification = replay.run_replay(source, output)

            self.assertEqual(verification["status"], "ok")
            self.assertEqual(verification["bootstrap"], {"samples": 40, "seed": 7})
            self.assertEqual(verification["secondary_comparisons"], 0)
            self.assertTrue((output / "summaries.json").is_file())
            self.assertTrue((output / "comparisons.json").is_file())
            self.assertTrue((output / "verification.json").is_file())
            written = json.loads(
                (output / "summaries.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("bases", written["summaries"]["paired_s17"])

    def test_condition_lists_compare_membership_only(self) -> None:
        comparisons = {"comparisons": [], "secondary_compositions": []}
        released_summaries = {
            "descriptive_control_conditions": ["fp", "constant_display_A"],
            "untrained_rule_conditions": ["constant_display_A"],
            "secondary_postprocessing_conditions": ["rtn_bias"],
        }
        regenerated_summaries = {
            "descriptive_control_conditions": ["constant_display_A", "fp"],
            "untrained_rule_conditions": ["constant_display_A"],
            "secondary_postprocessing_conditions": ["rtn_bias"],
        }
        replay.verify_report(
            released_summaries,
            comparisons,
            regenerated_summaries,
            comparisons,
        )

        missing = copy.deepcopy(regenerated_summaries)
        missing["descriptive_control_conditions"] = ["constant_display_A"]
        duplicate = copy.deepcopy(regenerated_summaries)
        duplicate["descriptive_control_conditions"].append("fp")
        for label, mutated in (("missing", missing), ("duplicate", duplicate)):
            with self.subTest(label=label):
                with self.assertRaises(replay.ReplayMismatch):
                    replay.verify_report(
                        released_summaries,
                        comparisons,
                        mutated,
                        comparisons,
                    )

    def test_tampered_secondary_composition_metric_fails(self) -> None:
        predictions = composition_predictions()
        recomputed = replay.recompute(
            predictions,
            condition_index=CONDITION_INDEX,
            bootstrap_samples=40,
            bootstrap_seed=7,
            model_id="synthetic/model",
        )
        self.assertTrue(recomputed["secondary_compositions"])
        released_summaries = {
            "format": "eq_quant_study_report_v1",
            "state": "completed",
            "model": {"id": "synthetic/model"},
            "conditions": recomputed["conditions"],
            "condition_index": copy.deepcopy(CONDITION_INDEX),
            "summaries": recomputed["summaries"],
            "grouped_summaries": recomputed["grouped_summaries"],
            "per_transform": recomputed["per_transform"],
            "descriptive_control_conditions": recomputed[
                "descriptive_control_conditions"
            ],
            "untrained_rule_conditions": recomputed[
                "untrained_rule_conditions"
            ],
        }
        released_comparisons = {
            "format": "eq_quant_study_report_v1",
            "state": "completed",
            "model": {"id": "synthetic/model"},
            "primary": "paired_vs_augmented_kd_same_seed",
            "evidence": "held_out_singles",
            "reference_condition": "fp",
            "bootstrap": {"unit": "base_id", "samples": 40, "seed": 7},
            "comparisons": recomputed["comparisons"],
            "secondary_compositions": recomputed["secondary_compositions"],
        }
        tampered = copy.deepcopy(released_comparisons)
        entry = find_comparison(
            tampered["secondary_compositions"], split="all", family="all"
        )
        entry["comparison"]["metrics"]["composition_accuracy"]["delta"] += 1e-6

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            write_source(
                source,
                predictions,
                released_summaries,
                tampered,
            )
            with self.assertRaises(replay.ReplayMismatch):
                replay.run_replay(source, output)
            self.assertFalse(output.exists())

    def test_tampered_released_metric_fails_without_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            tampered = copy.deepcopy(self.released_comparisons)
            entry = find_comparison(
                tampered["comparisons"], split="all", family="all"
            )
            entry["comparison"]["metrics"]["held_out_accuracy"]["delta"] += 1e-6
            write_source(
                source,
                self.predictions,
                self.released_summaries,
                tampered,
                self.selection,
            )

            with self.assertRaises(replay.ReplayMismatch):
                replay.run_replay(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
