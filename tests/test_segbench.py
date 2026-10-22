"""Segmentation benchmark: ground-truth masks in, descriptor agreement out."""

import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.data import make_synthetic  # noqa: E402
from pilot.segbench import benchmark, summarise  # noqa: E402


class TestSegBench(unittest.TestCase):
    def test_benchmark_against_ground_truth_masks(self):
        import pandas as pd
        from PIL import Image, ImageDraw

        root = Path(__file__).resolve().parent / "_tmp_segbench"
        shutil.rmtree(root, ignore_errors=True)
        try:
            make_synthetic(root, per_class=1, n_cells=120, size=256, seed=5)
            masks = root / "masks"
            masks.mkdir(parents=True, exist_ok=True)
            for img in sorted((root / "images").glob("*.png")):
                cells = pd.read_csv(root / "cells" / f"{img.stem}.csv")
                canvas = Image.new("L", (256, 256), 0)
                draw = ImageDraw.Draw(canvas)
                for idx, (x, y) in enumerate(zip(cells["x"], cells["y"]), start=1):
                    draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=idx % 255)
                canvas.save(masks / img.name)

            df = benchmark(root / "images", masks, limit=3)
            s = summarise(df)
            self.assertGreater(s["n_patches"], 0)
            self.assertGreater(s["detection_f1"], 0.5)
            self.assertTrue(np.isfinite(s["descriptor_delta_median"]))
            self.assertIn("hole_agreement", s)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
