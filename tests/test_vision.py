import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.data import make_synthetic  # noqa: E402
from pilot.data import build_cf_examples  # noqa: E402


class TestVision(unittest.TestCase):
    def test_fit_image_caps_tokens(self):
        from PIL import Image

        from pilot.vision import estimate_tokens, fit_image, side_for_tokens

        side = side_for_tokens(512)
        img = Image.new("RGB", (1024, 768), (255, 255, 255))
        out = fit_image(img, 512)
        self.assertLessEqual(max(out.size), side)
        self.assertLessEqual(estimate_tokens(*out.size), 560)
        small = Image.new("RGB", (100, 80), (0, 0, 0))
        self.assertIs(fit_image(small, 512), small)

    def test_cf_examples_have_location(self):
        root = Path(__file__).resolve().parent / "_tmp_cf"
        shutil.rmtree(root, ignore_errors=True)
        try:
            df = make_synthetic(root, per_class=1, n_cells=140, size=256, seed=3)
            rows = build_cf_examples(df, root / "cells", per_roi=2, seed=0)
            self.assertTrue(rows)
            for r in rows:
                self.assertIn("center", r)
                self.assertIn("radius", r)
                self.assertEqual(len(r["center"]), 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
