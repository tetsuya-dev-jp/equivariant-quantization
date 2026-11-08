import unittest

import torch
from torch import nn

from eq_quant.evaluate.quantize import (
    apply_rtn_to_model,
    quantizable_linear_modules,
    quantize_weight_rtn,
    restore_linear_weights,
    snapshot_linear_weights,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layer = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 8)


class QuantizeTests(unittest.TestCase):
    def test_zero_group_is_exact_zero(self) -> None:
        weight = torch.zeros(2, 4, dtype=torch.float32)
        result = quantize_weight_rtn(weight, bits=4, group_size=2)
        self.assertTrue(torch.equal(result.scales, torch.zeros_like(result.scales)))
        self.assertTrue(torch.equal(result.codes, torch.zeros_like(result.codes)))
        self.assertTrue(
            torch.equal(result.dequantized, torch.zeros_like(result.dequantized))
        )
        self.assertFalse(torch.isnan(result.dequantized).any())

    def test_tail_group_shapes_and_arithmetic(self) -> None:
        weight = torch.tensor(
            [[0.0, 1.0, -2.0, 0.5, 3.0]], dtype=torch.float32
        )
        result = quantize_weight_rtn(weight, bits=4, group_size=2)
        self.assertEqual(result.scales.shape, (1, 3))
        self.assertEqual(result.codes.shape, (1, 5))
        self.assertEqual(result.dequantized.shape, (1, 5))
        for group in range(3):
            start = group * 2
            end = min(start + 2, 5)
            expected = (
                result.codes[:, start:end].to(torch.float32)
                * result.scales[:, group : group + 1]
            )
            self.assertTrue(
                torch.equal(result.dequantized[:, start:end], expected)
            )

    def test_rounding_and_clamping_range(self) -> None:
        weight = torch.tensor(
            [[-3.0, -1.0, 1.0, 3.0]], dtype=torch.float32
        )
        result = quantize_weight_rtn(weight, bits=4, group_size=4)
        self.assertEqual(result.codes.tolist(), [[-7, -2, 2, 7]])
        self.assertGreaterEqual(int(result.codes.min()), -8)
        self.assertLessEqual(int(result.codes.max()), 7)

        weight2 = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
        result2 = quantize_weight_rtn(weight2, bits=4, group_size=4)
        self.assertEqual(result2.codes.tolist(), [[0, 2, 5, 7]])

    def test_idempotent_reconstruction(self) -> None:
        weight = torch.tensor(
            [[0.0, 0.5, -0.5, 1.0, -1.0, 0.25]], dtype=torch.float32
        )
        first = quantize_weight_rtn(weight, bits=8, group_size=4)
        second = quantize_weight_rtn(first.dequantized, bits=8, group_size=4)
        self.assertTrue(torch.equal(first.dequantized, second.dequantized))

    def test_excluded_modules(self) -> None:
        model = TinyModel()
        modules = quantizable_linear_modules(model)
        self.assertEqual([name for name, _ in modules], ["layer"])

    def test_apply_and_restore(self) -> None:
        torch.manual_seed(0)
        model = TinyModel()
        modules = quantizable_linear_modules(model)
        snapshot = snapshot_linear_weights(modules)
        head_before = model.lm_head.weight.detach().clone()
        original = model.layer.weight.detach().clone()

        inventory = apply_rtn_to_model(model, bits=4, group_size=2)
        self.assertEqual(len(inventory), 1)
        entry = inventory[0]
        self.assertEqual(entry["name"], "layer")
        self.assertEqual(entry["bits"], 4)
        self.assertLessEqual(entry["code_min"], entry["code_max"])
        self.assertGreaterEqual(entry["code_min"], -8)
        self.assertLessEqual(entry["code_max"], 7)
        self.assertFalse(torch.equal(model.layer.weight, original))
        self.assertTrue(torch.equal(model.lm_head.weight, head_before))

        restore_linear_weights(modules, snapshot)
        self.assertTrue(torch.equal(model.layer.weight, original))
        self.assertTrue(torch.equal(model.lm_head.weight, head_before))


if __name__ == "__main__":
    unittest.main()
