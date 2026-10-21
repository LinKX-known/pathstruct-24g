import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.data import (  # noqa: E402
    build_cf_examples,
    build_growth_examples,
    make_synthetic,
    read_jsonl,
    read_manifest,
    write_jsonl,
)
from pilot.eval import normalise, paired_bootstrap, parse, score  # noqa: E402
from pilot.structures import topology_summary  # noqa: E402


class TestSyntheticPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent / "_tmp_pilot"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.manifest = make_synthetic(cls.root / "data", per_class=3, n_cells=140, size=128, seed=0)
        # a larger cohort for the pixel -> cell -> topology test: at 128 px the
        # synthetic dots overlap too much for the classical segmenter to be stable
        cls.seg_root = Path(__file__).resolve().parent / "_tmp_seg"
        shutil.rmtree(cls.seg_root, ignore_errors=True)
        make_synthetic(cls.seg_root, per_class=1, n_cells=200, size=256, seed=7)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)
        shutil.rmtree(cls.seg_root, ignore_errors=True)

    def test_manifest_and_images(self):
        df = read_manifest(self.root / "data" / "manifest.csv")
        self.assertEqual(len(df), 9)
        for p in df["image_path"]:
            self.assertTrue(Path(p).exists())

    def test_growth_examples(self):
        df = read_manifest(self.root / "data" / "manifest.csv")
        ex = build_growth_examples(df)
        self.assertEqual(len(ex), 9)
        self.assertTrue(all(e["answer"] in ("ring", "solid", "multifocal") for e in ex))

    def test_cf_examples_are_balanced_and_labelled(self):
        df = read_manifest(self.root / "data" / "manifest.csv")
        ex = build_cf_examples(df, self.root / "data" / "cells", per_roi=4, seed=1)
        self.assertGreater(len(ex), 0)
        rate = float(np.mean([e["changed"] for e in ex]))
        self.assertGreater(rate, 0.15)
        self.assertLess(rate, 0.85)
        for e in ex:
            self.assertIn(e["answer"], ("yes", "no"))
            self.assertIn("normalised 0-1000", e["question"])
            self.assertEqual(len(e["descriptor"]), len(ex[0]["descriptor"]))

    def test_jsonl_roundtrip(self):
        df = read_manifest(self.root / "data" / "manifest.csv")
        ex = build_growth_examples(df)
        path = self.root / "growth.jsonl"
        self.assertEqual(write_jsonl(ex, path), len(ex))
        self.assertEqual(len(read_jsonl(path)), len(ex))

    def test_topology_labels(self):
        ring = topology_summary(self._points("ring"))
        solid = topology_summary(self._points("solid"))
        self.assertEqual(ring["hole_count"], 1.0)
        self.assertEqual(solid["hole_count"], 0.0)

    def test_classical_segmentation_recovers_arrangement(self):
        try:
            from pilot.segment import segment_image
        except ImportError:
            self.skipTest("scikit-image not installed")
        images = self.seg_root / "images"
        ring_pts, _ = segment_image(images / "ring_000.png")
        solid_pts, _ = segment_image(images / "solid_000.png")
        multi_pts, _ = segment_image(images / "multifocal_000.png")
        self.assertGreater(len(ring_pts), 20)
        self.assertEqual(topology_summary(ring_pts)["hole_count"], 1.0)
        self.assertEqual(topology_summary(solid_pts)["hole_count"], 0.0)
        self.assertEqual(topology_summary(multi_pts)["n_comp"], 3.0)

    def _points(self, kind):
        rng = np.random.default_rng(0)
        t = rng.uniform(0, 2 * np.pi, 80)
        if kind == "ring":
            r = 0.7 + rng.normal(scale=0.01, size=80)
        else:
            r = np.sqrt(rng.uniform(0, 0.7**2, 80))
        return np.c_[r * np.cos(t), r * np.sin(t)]


class TestEvalHelpers(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse("acinar", ["acinar", "solid"]), "acinar")
        self.assertEqual(parse("The answer is Solid.", ["acinar", "solid"]), "solid")
        self.assertIsNone(parse("no idea", ["yes", "no"]))
    def test_normalise(self):
        self.assertEqual(normalise("  Yes. "), "yes")

    def test_score_and_bootstrap(self):
        rows = [
            {"gold": "a", "pred": "a", "correct": 1},
            {"gold": "b", "pred": "b", "correct": 1},
            {"gold": "a", "pred": "b", "correct": 0},
        ]
        s = score(rows)
        self.assertAlmostEqual(s["accuracy"], 2 / 3)
        out = paired_bootstrap([1, 1, 0], [1, 0, 0])
        self.assertAlmostEqual(out["delta"], 1 / 3)
        self.assertLessEqual(out["lo"], out["delta"])
        self.assertGreaterEqual(out["hi"], out["delta"])


class TestSegBench(unittest.TestCase):
    def test_benchmark_against_ground_truth_masks(self):
        import pandas as pd
        from PIL import Image, ImageDraw

        from pilot.segbench import benchmark, summarise

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
