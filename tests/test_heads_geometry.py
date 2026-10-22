import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

if HAS_TORCH:
    from pilot.heads import (  # noqa: E402
        CounterfactualHead,
        grid_from_hidden,
        grid_or_global_mean,
    )


@unittest.skipUnless(HAS_TORCH, "torch is required")
class TestGridFallback(unittest.TestCase):
    def test_grid_when_shapes_match(self):
        hidden = torch.arange(1 * 4 * 8, dtype=torch.float32).reshape(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        thw = torch.tensor([[1, 2, 2]])
        grid, status = grid_or_global_mean(hidden, mask, thw, merge_size=1)
        self.assertEqual(status, "grid")
        self.assertEqual(tuple(grid.shape), (2, 2, 8))
        self.assertTrue(torch.equal(grid, grid_from_hidden(hidden, mask, thw, merge_size=1)))

    def test_shape_mismatch_falls_back_and_is_reported(self):
        hidden = torch.arange(1 * 4 * 8, dtype=torch.float32).reshape(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        thw = torch.tensor([[1, 3, 3]])  # expects 9 tokens, not 4
        grid, status = grid_or_global_mean(hidden, mask, thw, merge_size=1)
        self.assertEqual(status, "fallback")
        self.assertEqual(tuple(grid.shape), (3, 3, 8))
        expected = hidden[0].mean(0)
        for i in range(3):
            for j in range(3):
                self.assertTrue(torch.allclose(grid[i, j], expected))

    def test_no_image_tokens_fails(self):
        hidden = torch.zeros(1, 4, 8)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        grid, status = grid_or_global_mean(hidden, mask, torch.tensor([[1, 2, 2]]))
        self.assertIsNone(grid)
        self.assertEqual(status, "failed")


@unittest.skipUnless(HAS_TORCH, "torch is required")
class TestCounterfactualGeometry(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.head = CounterfactualHead(d_model=16, hidden=32)
        self.grid = torch.randn(3, 3, 16)

    def _feats(self, cx=0.5, cy=0.5, radius=0.2):
        return torch.tensor([[cx, cy, radius, 0.0, 0.0, 1.0, 0.0, 0.0]])

    def test_valid_normalised_geometry(self):
        out = self.head(self.grid, self._feats())
        self.assertEqual(tuple(out.shape), (1,))

    def test_pixel_centre_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "normalised"):
            self.head(self.grid, self._feats(cx=512.0, cy=400.0))

    def test_pixel_radius_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "radius"):
            self.head(self.grid, self._feats(radius=60.0))

    def test_nan_geometry_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.head(self.grid, self._feats(cx=float("nan")))


@unittest.skipUnless(HAS_TORCH, "torch is required")
class TestAuxCollation(unittest.TestCase):
    def test_growth_row_without_changed(self):
        from pilot.train import collate_aux

        ex = {
            "descriptor": [0.1, 0.2, 0.3, 0.4, 0.5],
            "center": [0.5, 0.5],
            "radius": 0.2,
            "kind": "collapse",
        }
        desc, changed, feats = collate_aux(ex)
        self.assertEqual(tuple(desc.shape), (1, 5))
        self.assertEqual(float(changed.item()), 0.0)
        self.assertEqual(tuple(feats.shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
