import unittest

import numpy as np

from pilot.hurdle import (
    expected_log1p,
    positive_class_weights,
    zero_inflated_dims,
)


class ZeroInflatedDimsTest(unittest.TestCase):
    def test_selects_only_the_zero_spike(self):
        rows = [
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.2],
            [5.0, 0.0, 0.0],
            [6.0, 1.0, 0.0],
        ]
        idx, zero_frac = zero_inflated_dims(rows, dims=3, threshold=0.5)
        self.assertEqual(idx, [1, 2])
        np.testing.assert_allclose(zero_frac, [0.0, 0.75, 0.75])

    def test_dimension_is_not_selected_below_threshold(self):
        rows = [[1.0, 0.0], [2.0, 0.1], [3.0, 0.2]]
        idx, _ = zero_inflated_dims(rows, dims=2, threshold=0.5)
        self.assertEqual(idx, [])

    def test_empty_pool_is_safe(self):
        idx, zero_frac = zero_inflated_dims([], dims=3)
        self.assertEqual(idx, [])
        self.assertEqual(zero_frac, [0.0, 0.0, 0.0])


class PositiveWeightsTest(unittest.TestCase):
    def test_inverse_frequency_and_clip(self):
        rows = [[0.0], [1.0]] * 9  # 50% positive -> weight 1
        self.assertEqual(positive_class_weights(rows, [0]), [1.0])
        rows = [[0.0]] * 19 + [[1.0]]  # 5% positive -> 19
        self.assertAlmostEqual(positive_class_weights(rows, [0])[0], 19.0)
        rows = [[0.0]] * 199 + [[1.0]]  # 0.5% positive -> clipped at 20
        self.assertEqual(positive_class_weights(rows, [0])[0], 20.0)


class ExpectedLog1pTest(unittest.TestCase):
    def test_presence_scales_only_hurdle_dims(self):
        out = expected_log1p([1.0, 2.0, 3.0], [0.5, 0.25], [1, 2])
        np.testing.assert_allclose(out, [1.0, 1.0, 0.75])

    def test_no_presence_is_identity(self):
        out = expected_log1p([1.0, 2.0], None, [0])
        np.testing.assert_allclose(out, [1.0, 2.0])


try:
    import torch

    from pilot.heads import build_heads

    class HeadWidthTest(unittest.TestCase):
        def test_presence_outputs_are_appended(self):
            heads = build_heads(16, 5, hidden=32, presence_dims=2)
            grid = torch.randn(3, 3, 16)
            self.assertEqual(tuple(heads["structure"](grid).shape), (1, 7))

except ImportError:  # pragma: no cover
    pass


if __name__ == "__main__":
    unittest.main()
