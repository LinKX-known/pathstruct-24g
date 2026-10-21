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


if __name__ == "__main__":
    unittest.main()

class TestCellVitBridge(unittest.TestCase):
    def test_parse_cellvit_json(self):
        import json as _json

        from pilot.cells import cells_from_cellvit_dir, cells_from_cellvit_json

        root = Path(__file__).resolve().parent / "_tmp_cellvit"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        try:
            payload = {
                "type_map": {"0": "Tumor", "1": "Stroma"},
                "cells": [
                    {"centroid": [10.5, 20.5], "class": "Tumor"},
                    {"centroid": [30.0, 40.0], "class": "Stroma"},
                    {"centroid": [5.0, 5.0], "class": "Tumor"},
                ],
            }
            (root / "roi_1_cells.json").write_text(_json.dumps(payload), encoding="utf-8")
            pts, types = cells_from_cellvit_json(root / "roi_1_cells.json")
            self.assertEqual(len(pts), 3)
            self.assertEqual(len(set(types.tolist())), 2)
            df = cells_from_cellvit_dir(root, root / "cells")
            self.assertEqual(int(df.iloc[0]["n_cells"]), 3)
            self.assertTrue((root / "cells" / "roi_1.csv").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_grouped_bootstrap_and_bh(self):
        from pilot.eval import benjamini_hochberg, paired_bootstrap

        a = [1, 1, 0, 1, 0, 1]
        b = [1, 0, 0, 0, 0, 1]
        groups = ["p1", "p1", "p1", "p2", "p2", "p2"]
        out = paired_bootstrap(a, b, n_boot=200, groups=groups)
        self.assertAlmostEqual(out["delta"], 0.3333, places=3)
        self.assertEqual(out["n_groups"], 2)
        adj = benjamini_hochberg([0.01, 0.04, 0.03])
        self.assertEqual(len(adj), 3)
        self.assertTrue(all(x <= 1.0 for x in adj))
        self.assertLess(adj[0], adj[1])

class TestVisionAndExamples(unittest.TestCase):
    def test_fit_image_caps_tokens(self):
        from PIL import Image

        from pilot.vision import estimate_tokens, fit_image, side_for_tokens

        side = side_for_tokens(512)
        img = Image.new("RGB", (1024, 768), (255, 255, 255))
        out = fit_image(img, 512)
        self.assertLessEqual(max(out.size), side)
        self.assertLessEqual(estimate_tokens(*out.size), 560)

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
