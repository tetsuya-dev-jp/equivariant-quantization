import json
import unittest
from collections import defaultdict
from pathlib import Path

from eq_quant.data.build import (
    SemanticCollisionError,
    build_dataset,
    register_semantic_key,
)
from eq_quant.data.generate import semantic_key
from eq_quant.data.oracles import boolean_value, canonical_answer
from eq_quant.data.render import render_prompt
from eq_quant.data.schema import (
    BASE_TRANSFORMS,
    BASE_TRANSFORM_AVAILABILITY,
    COMPOSED_SPLITS,
    COMPOSED_TRANSFORMS,
    COMPOSED_TRANSFORM_AVAILABILITY,
    TRANSFORM_AVAILABILITY,
    BaseInstance,
    available_transforms,
)

OPTION_PERMUTED_TRANSFORMS = {
    "option_permutation",
    "option_permutation+entity_rename",
    "option_permutation+entity_rename+order_permutation",
}
COMPOSED_OPTION_RENAME = "option_permutation+entity_rename"
COMPOSED_OPTION_RENAME_ORDER = (
    "option_permutation+entity_rename+order_permutation"
)


def option_lines(prompt: str) -> dict[str, str]:
    lines: dict[str, str] = {}
    for line in prompt.splitlines():
        if len(line) >= 3 and line[0] in "ABCD" and line[1] == ".":
            lines[line[0]] = line[3:]
    return lines


def small_config() -> dict:
    return {
        "data": {
            "families": ["arithmetic", "boolean", "graph"],
            "splits": {
                "calibration": {"count": 5, "seed": 1108},
                "development": {"count": 5, "seed": 2108},
                "pilot": {"count": 4, "seed": 3108},
                "final": {"count": 6, "seed": 4108},
                "structural_holdout": {"count": 4, "seed": 5108},
            },
        }
    }


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = small_config()
        cls.dataset = build_dataset(
            cls.config,
            split_names=["calibration", "pilot", "structural_holdout"],
        )
        cls.records_by_base = defaultdict(list)
        for record in cls.dataset.records:
            cls.records_by_base[record.base_id].append(record)

    def test_oracles_and_distinct_canonical_options(self) -> None:
        for base in self.dataset.bases:
            correct = canonical_answer(
                base.task_family, base.structure, base.canonical_options
            )
            self.assertIsInstance(correct, int)
            self.assertGreaterEqual(correct, 0)
            self.assertLess(correct, 4)
            if base.task_family == "arithmetic":
                self.assertEqual(len(set(base.canonical_options)), 4)
            elif base.task_family == "boolean":
                as_tuples = [tuple(option) for option in base.canonical_options]
                self.assertEqual(len(set(as_tuples)), 4)
                true_count = 0
                for option in base.canonical_options:
                    assignment = dict(
                        zip(
                            base.structure["variables"],
                            option,
                            strict=True,
                        )
                    )
                    if boolean_value(base.structure["expression"], assignment):
                        true_count += 1
                self.assertEqual(true_count, 1)
            else:
                self.assertEqual(len(set(base.canonical_options)), 4)

    def test_permutation_mapping_direction(self) -> None:
        for record in self.dataset.records:
            self.assertEqual(sorted(record.option_order), [0, 1, 2, 3])
            correct = canonical_answer(
                record.task_family,
                record.semantic["structure"],
                record.semantic["canonical_options"],
            )
            self.assertEqual(
                record.correct_option_index, record.option_order.index(correct)
            )
            if record.transform in OPTION_PERMUTED_TRANSFORMS:
                self.assertNotEqual(record.option_order, [0, 1, 2, 3])
            else:
                self.assertEqual(record.option_order, [0, 1, 2, 3])

            for display_index, canonical_index in enumerate(record.option_order):
                self.assertIn(canonical_index, range(4))
                # The displayed slot's canonical identity is option_order[slot].
                self.assertIn(f"{'ABCD'[display_index]}.", record.prompt)

    def test_renderer_option_mapping_direction(self) -> None:
        permutation = [2, 0, 3, 1]
        for base in self.dataset.bases:
            identity_prompt = render_prompt(
                base.task_family,
                base.structure,
                base.canonical_options,
                [0, 1, 2, 3],
                None,
                None,
            )
            permuted_prompt = render_prompt(
                base.task_family,
                base.structure,
                base.canonical_options,
                permutation,
                None,
                None,
            )
            identity_lines = option_lines(identity_prompt)
            permuted_lines = option_lines(permuted_prompt)
            self.assertEqual(sorted(identity_lines), list("ABCD"))
            self.assertEqual(sorted(permuted_lines), list("ABCD"))
            for display_index, canonical_index in enumerate(permutation):
                display_label = "ABCD"[display_index]
                canonical_label = "ABCD"[canonical_index]
                self.assertEqual(
                    permuted_lines[display_label],
                    identity_lines[canonical_label],
                )

    def test_transform_availability_and_semantics(self) -> None:
        for base in self.dataset.bases:
            records = self.records_by_base[base.base_id]
            self.assertEqual(
                tuple(record.transform for record in records),
                available_transforms(base.split, base.task_family),
            )
            semantic_structures = {
                repr(record.semantic["structure"]) for record in records
            }
            semantic_options = {
                repr(record.semantic["canonical_options"]) for record in records
            }
            self.assertEqual(len(semantic_structures), 1)
            self.assertEqual(len(semantic_options), 1)

            correct = canonical_answer(
                base.task_family, base.structure, base.canonical_options
            )
            for record in records:
                self.assertEqual(
                    canonical_answer(
                        record.task_family,
                        record.semantic["structure"],
                        record.semantic["canonical_options"],
                    ),
                    correct,
                )

            for record in records:
                if record.rename is not None:
                    rename = record.rename["canonical_to_display"]
                    self.assertEqual(len(rename), len(set(rename.values())))
                    for old_name, new_name in rename.items():
                        self.assertNotEqual(old_name, new_name)
                    if base.task_family == "arithmetic":
                        old_name = base.structure["name"]
                        self.assertNotIn(old_name, record.prompt)
                        self.assertIn(rename[old_name], record.prompt)
                    elif base.task_family == "boolean":
                        for new_name in rename.values():
                            self.assertIn(new_name, record.prompt)
                    else:
                        for left, right in base.structure["edges"]:
                            self.assertIn(
                                f"{rename[left]}->{rename[right]}",
                                record.prompt,
                            )
                        self.assertIn(
                            rename[base.structure["source"]], record.prompt
                        )
                if record.order is not None:
                    order_values = list(record.order.values())[0]
                    self.assertGreater(len(order_values), 1)
                    self.assertNotEqual(
                        order_values, list(range(len(order_values)))
                    )
                if record.transform in ("original", "option_permutation"):
                    self.assertIsNone(record.rename)
                    self.assertIsNone(record.order)
                if record.transform == COMPOSED_OPTION_RENAME:
                    self.assertIsNotNone(record.rename)
                    self.assertIsNone(record.order)
                if record.transform == COMPOSED_OPTION_RENAME_ORDER:
                    self.assertIsNotNone(record.rename)
                    self.assertIsNotNone(record.order)

    def test_split_disjointness_and_determinism(self) -> None:
        keys = defaultdict(set)
        for base in self.dataset.bases:
            keys[base.split].add(base.semantic_key)
        splits = sorted(keys)
        for left_index, left in enumerate(splits):
            for right in splits[left_index + 1 :]:
                self.assertTrue(keys[left].isdisjoint(keys[right]))

        rebuilt = build_dataset(
            self.config,
            split_names=["calibration", "pilot", "structural_holdout"],
        )
        self.assertEqual(
            [base.to_dict() for base in self.dataset.bases],
            [base.to_dict() for base in rebuilt.bases],
        )
        self.assertEqual(
            [record.to_dict() for record in self.dataset.records],
            [record.to_dict() for record in rebuilt.records],
        )
        self.assertEqual(self.dataset.manifest, rebuilt.manifest)

        record_ids = [record.record_id for record in self.dataset.records]
        self.assertEqual(len(record_ids), len(set(record_ids)))

    def test_collision_rejection(self) -> None:
        seen: dict[str, str] = {}
        register_semantic_key(seen, "arithmetic|same", "calibration")
        with self.assertRaises(SemanticCollisionError):
            register_semantic_key(seen, "arithmetic|same", "pilot")

    def test_structural_key_ignores_options_and_proper_names(self) -> None:
        structure = {
            "name": "Alice",
            "expression": {"op": "+", "left": 2, "right": 3},
            "operation_count": 1,
        }
        renamed_only = {**structure, "name": "Ben"}
        key_a = semantic_key("arithmetic", structure)
        key_b = semantic_key("arithmetic", renamed_only)
        self.assertEqual(key_a, key_b)

        base_a = BaseInstance(
            base_id="arithmetic-calibration-0000",
            split="calibration",
            task_family="arithmetic",
            structure=structure,
            canonical_options=[5, 9, 8, 7],
            semantic_key=key_a,
        )
        base_b = BaseInstance(
            base_id="arithmetic-pilot-0000",
            split="pilot",
            task_family="arithmetic",
            structure=renamed_only,
            canonical_options=[5, 1, 2, 4],
            semantic_key=key_b,
        )
        self.assertEqual(base_a.semantic_key, base_b.semantic_key)

        seen: dict[str, str] = {}
        register_semantic_key(seen, base_a.semantic_key, base_a.split)
        with self.assertRaises(SemanticCollisionError):
            register_semantic_key(seen, base_b.semantic_key, base_b.split)

    def test_subset_generation_matches_full_generation(self) -> None:
        full = build_dataset(self.config)
        splits = (
            "calibration",
            "development",
            "pilot",
            "final",
            "structural_holdout",
        )
        for split in splits:
            subset = build_dataset(self.config, split_names=[split])
            self.assertEqual(
                [base.to_dict() for base in subset.bases],
                [
                    base.to_dict()
                    for base in full.bases
                    if base.split == split
                ],
            )
            self.assertEqual(
                [record.to_dict() for record in subset.records],
                [
                    record.to_dict()
                    for record in full.records
                    if record.split == split
                ],
            )

        default = build_dataset(self.config)
        self.assertEqual(
            [record.to_dict() for record in default.records],
            [record.to_dict() for record in full.records],
        )
        self.assertEqual(default.manifest, full.manifest)

    def test_full_default_build(self) -> None:
        config = json.loads(
            Path("configs/pilot.json").read_text(encoding="utf-8")
        )
        dataset = build_dataset(config)
        self.assertEqual(len(dataset.bases), 2112)
        self.assertEqual(len(dataset.records), 9664)
        self.assertEqual(
            len({base.semantic_key for base in dataset.bases}), 2112
        )
        self.assertEqual(
            dataset.manifest["counts"]["per_split"]["pilot"]["bases"], 192
        )
        composed = [
            record
            for record in dataset.records
            if record.transform in COMPOSED_TRANSFORMS
        ]
        self.assertEqual(len(composed), 1920)
        self.assertTrue(
            all(
                record.split in COMPOSED_SPLITS
                for record in composed
            )
        )
        self.assertEqual(
            sum(
                1
                for record in dataset.records
                if record.transform in BASE_TRANSFORMS
            ),
            7744,
        )

        rebuilt = build_dataset(config)
        self.assertEqual(
            [record.to_dict() for record in rebuilt.records],
            [record.to_dict() for record in dataset.records],
        )
        self.assertEqual(rebuilt.manifest, dataset.manifest)

    def test_composed_records_only_in_final_and_structural_holdout(self) -> None:
        dataset = build_dataset(self.config)
        records_by_base = defaultdict(list)
        for record in dataset.records:
            records_by_base[record.base_id].append(record)

        for base in dataset.bases:
            self.assertEqual(
                tuple(
                    record.transform
                    for record in records_by_base[base.base_id]
                ),
                available_transforms(base.split, base.task_family),
            )

        composed = [
            record
            for record in dataset.records
            if record.transform in COMPOSED_TRANSFORMS
        ]
        self.assertTrue(composed)
        for record in composed:
            self.assertIn(record.split, COMPOSED_SPLITS)
        for record in dataset.records:
            if record.transform not in COMPOSED_TRANSFORMS:
                self.assertIn(record.transform, BASE_TRANSFORMS)
            if record.split not in COMPOSED_SPLITS:
                self.assertNotIn(record.transform, COMPOSED_TRANSFORMS)

    def test_composed_option_correspondence_and_component_reuse(self) -> None:
        dataset = build_dataset(self.config)
        records_by_base = defaultdict(dict)
        for record in dataset.records:
            records_by_base[record.base_id][record.transform] = record

        non_self_inverse = []
        for base in dataset.bases:
            if base.split not in COMPOSED_SPLITS:
                continue
            components = records_by_base[base.base_id]
            correct = canonical_answer(
                base.task_family, base.structure, base.canonical_options
            )
            expected_semantic = {
                "structure": base.structure,
                "canonical_options": base.canonical_options,
            }
            for transform in COMPOSED_TRANSFORM_AVAILABILITY[base.task_family]:
                record = components[transform]
                option_record = components["option_permutation"]
                rename_record = components["entity_rename"]

                self.assertEqual(
                    record.option_order, option_record.option_order
                )
                self.assertEqual(record.rename, rename_record.rename)
                if transform == COMPOSED_OPTION_RENAME:
                    self.assertIsNone(record.order)
                else:
                    order_record = components["order_permutation"]
                    self.assertEqual(record.order, order_record.order)

                self.assertEqual(record.semantic, expected_semantic)
                self.assertEqual(
                    record.correct_option_index,
                    record.option_order.index(correct),
                )
                self.assertEqual(
                    record.prompt,
                    render_prompt(
                        base.task_family,
                        base.structure,
                        base.canonical_options,
                        record.option_order,
                        record.rename,
                        record.order,
                    ),
                )

                permutation = record.option_order
                twice = [
                    permutation[permutation[index]]
                    for index in range(len(permutation))
                ]
                if twice != list(range(len(permutation))):
                    non_self_inverse.append(record)

        # The correspondence must hold for permutations that are not
        # self-inverse, not only for swaps.
        self.assertTrue(non_self_inverse)
        for record in non_self_inverse:
            self.assertNotEqual(record.option_order, [0, 1, 2, 3])
            correct = canonical_answer(
                record.task_family,
                record.semantic["structure"],
                record.semantic["canonical_options"],
            )
            self.assertEqual(
                record.correct_option_index,
                record.option_order.index(correct),
            )

    def test_arithmetic_option_rename_order_composition_unavailable(self) -> None:
        self.assertNotIn(
            COMPOSED_OPTION_RENAME_ORDER,
            COMPOSED_TRANSFORM_AVAILABILITY["arithmetic"],
        )
        self.assertNotIn(
            COMPOSED_OPTION_RENAME_ORDER,
            TRANSFORM_AVAILABILITY["arithmetic"],
        )
        for split in COMPOSED_SPLITS:
            self.assertNotIn(
                COMPOSED_OPTION_RENAME_ORDER,
                available_transforms(split, "arithmetic"),
            )
            self.assertIn(
                COMPOSED_OPTION_RENAME,
                available_transforms(split, "arithmetic"),
            )

        dataset = build_dataset(self.config)
        for record in dataset.records:
            if record.task_family == "arithmetic":
                self.assertNotEqual(
                    record.transform, COMPOSED_OPTION_RENAME_ORDER
                )

    def test_non_composed_records_unchanged(self) -> None:
        dataset = build_dataset(self.config)
        old_splits = ("calibration", "development", "pilot")
        unchanged = [
            record.to_dict()
            for record in dataset.records
            if record.split in old_splits
        ]
        rebuilt = build_dataset(self.config, split_names=list(old_splits))
        self.assertEqual(
            [record.to_dict() for record in rebuilt.records], unchanged
        )
        for record in rebuilt.records:
            self.assertIn(record.transform, BASE_TRANSFORMS)
        expected = sum(
            len(BASE_TRANSFORM_AVAILABILITY[base.task_family])
            for base in dataset.bases
            if base.split in old_splits
        )
        self.assertEqual(len(unchanged), expected)

    def test_structural_holdout_conditions(self) -> None:
        normal_graph_sizes = set()
        holdout_graph_sizes = set()
        for base in self.dataset.bases:
            if base.task_family == "arithmetic":
                operation_count = base.structure["operation_count"]
                if base.split == "structural_holdout":
                    self.assertIn(operation_count, (5, 6))
                else:
                    self.assertIn(operation_count, (2, 3))
            elif base.task_family == "boolean":
                variable_count = base.structure["variable_count"]
                if base.split == "structural_holdout":
                    self.assertEqual(variable_count, 5)
                    self.assertEqual(base.structure["depth"], 3)
                else:
                    self.assertIn(variable_count, (3, 4))
                    self.assertEqual(base.structure["depth"], 2)
            else:
                node_count = len(base.structure["nodes"])
                if base.split == "structural_holdout":
                    self.assertIn(node_count, (8, 9, 10))
                    holdout_graph_sizes.add(node_count)
                else:
                    self.assertIn(node_count, (5, 6, 7))
                    normal_graph_sizes.add(node_count)
        self.assertTrue(holdout_graph_sizes.isdisjoint(normal_graph_sizes))


if __name__ == "__main__":
    unittest.main()
