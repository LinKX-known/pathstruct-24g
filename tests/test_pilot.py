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
        ex = build_cf_examples(
            df,
            self.root / "data" / "cells",
            per_roi=6,
            seed=1,
            annotate=True,
            annotate_dir=self.root / "data" / "annotated",
        )
        self.assertGreater(len(ex), 0)
        rate = float(np.mean([e["changed"] for e in ex]))
        self.assertGreater(rate, 0.15)
        self.assertLess(rate, 0.85)
        for e in ex:
            self.assertIn(e["answer"], ("yes", "no"))
            self.assertIn("green box", e["question"])
            self.assertIn("connected cell groups", e["question"])
            # The prompt must not contain coordinates: a text-only model must
            # not be able to learn a position prior.
            self.assertNotIn("[", e["question"])
            self.assertIn(e["kind"], ("collapse", "expand", "translate"))
            self.assertEqual(len(e["descriptor"]), len(ex[0]["descriptor"]))
            self.assertTrue(0.0 <= e["center"][0] <= 1.0)
            self.assertTrue(0.0 <= e["center"][1] <= 1.0)
            self.assertGreater(e["radius"], 0.0)
        # Balance within kind: a text-only model must not be able to read the
        # edit kind and guess the majority label.
        for kind in ("collapse", "expand", "translate"):
            labels = [e["changed"] for e in ex if e["kind"] == kind]
            self.assertEqual(sum(labels), len(labels) - sum(labels), kind)
        # translate prompts point at the visual arrow; the direction itself is
        # not text (otherwise a text model could read position/direction).
        self.assertTrue(
            all("orange arrow" in e["question"] for e in ex if e["kind"] == "translate")
        )

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


class TestTaskDesign(unittest.TestCase):
    def test_parse_verbose_binary(self):
        self.assertEqual(parse("Yes, because the ring closes.", ["yes", "no"]), "yes")
        self.assertEqual(parse("No, the sheet stays connected.", ["yes", "no"]), "no")
        self.assertIsNone(parse("no idea", ["yes", "no"]))
        self.assertIsNone(parse("There is not enough information.", ["yes", "no"]))

    def test_patient_splits_are_disjoint(self):
        from pilot.data import patient_splits

        root = Path(__file__).resolve().parent / "_tmp_split"
        shutil.rmtree(root, ignore_errors=True)
        try:
            df = make_synthetic(root, per_class=3, n_cells=140, size=128, seed=5)
            growth = build_growth_examples(df)
            cf = build_cf_examples(df, root / "cells", per_roi=2, seed=0)
            gtr, gval, ctr, cval = patient_splits(growth, cf, val_frac=0.4, seed=0)
            self.assertTrue(gtr and gval and ctr and cval)
            gtr_p = {r["patient_id"] for r in gtr}
            gval_p = {r["patient_id"] for r in gval}
            ctr_p = {r["patient_id"] for r in ctr}
            cval_p = {r["patient_id"] for r in cval}
            self.assertFalse(gtr_p & gval_p)
            self.assertFalse(ctr_p & cval_p)
            self.assertEqual(gval_p, cval_p)
            all_p = {r["patient_id"] for r in growth} | {r["patient_id"] for r in cf}
            self.assertEqual(gtr_p | gval_p, all_p)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_annotated_images_and_normalised_features(self):
        root = Path(__file__).resolve().parent / "_tmp_annot"
        shutil.rmtree(root, ignore_errors=True)
        try:
            df = make_synthetic(root, per_class=1, n_cells=140, size=256, seed=9)
            rows = build_cf_examples(
                df,
                root / "cells",
                per_roi=6,
                seed=0,
                annotate=True,
                annotate_dir=root / "annotated",
            )
            self.assertTrue(rows)
            for r in rows:
                self.assertNotEqual(r["image"], r["source_image"])
                self.assertTrue(Path(r["image"]).exists())
                self.assertEqual(len(r["vector"]), 2)
                self.assertEqual(len(r["box_1000"]), 4)
                self.assertEqual(len(r["image_size"]), 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_min_affected_filters_empty_boxes(self):
        from pilot.interventions import sample_interventions

        rng = np.random.default_rng(0)
        t = rng.uniform(0, 2 * np.pi, 120)
        r = 0.7 + rng.normal(scale=0.02, size=120)
        points = np.c_[r * np.cos(t), r * np.sin(t)]
        out = sample_interventions(points, n=6, seed=0, min_affected=5)
        self.assertTrue(out)
        self.assertTrue(all(i.edit.affected >= 5 for i in out))

    def test_prompt_matches_annotation_mode(self):
        root = Path(__file__).resolve().parent / "_tmp_prompt"
        shutil.rmtree(root, ignore_errors=True)
        try:
            df = make_synthetic(root, per_class=1, n_cells=140, size=256, seed=11)
            plain = build_cf_examples(df, root / "cells", per_roi=6, seed=0)
            self.assertTrue(plain)
            self.assertTrue(all("0-1000 image coordinates" in r["question"] for r in plain))
            self.assertTrue(all("green box" not in r["question"] for r in plain))
            self.assertTrue(all(not r["annotated"] for r in plain))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_regional_descriptor_uses_image_frame(self):
        from pilot.structures import regional_descriptor

        # (0,0) -> row 0 col 0; (99,0) -> row 0 col 2; (99,99) -> row 2 col 2.
        # Row-major indexing must map the middle point to tile 2, not tile 6.
        points = np.asarray([[0.0, 0.0], [99.0, 0.0], [99.0, 99.0]])
        types = np.zeros(3, dtype=int)
        out = regional_descriptor(
            points, types, n_types=1, grid=3, frame=((0.0, 0.0), (100.0, 100.0))
        )
        self.assertAlmostEqual(out[0], 1 / 3)
        self.assertAlmostEqual(out[2], 1 / 3)
        self.assertAlmostEqual(out[8], 1 / 3)
        self.assertAlmostEqual(out[9 + 2], 1 / 3)
        self.assertAlmostEqual(out[9 + 8], 1 / 3)

    def test_read_jsonl_accepts_json_array(self):
        from pilot.data import read_jsonl, write_jsonl

        root = Path(__file__).resolve().parent / "_tmp_json"
        shutil.rmtree(root, ignore_errors=True)
        try:
            root.mkdir(parents=True)
            path = root / "rows.json"
            path.write_text('[{"a": 1}, {"a": 2}]', encoding="utf-8")
            self.assertEqual(len(read_jsonl(path)), 2)
            rows = [{"a": 1}, {"a": 2}]
            write_jsonl(rows, root / "rows.jsonl")
            self.assertEqual(len(read_jsonl(root / "rows.jsonl")), 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestCellTypeVocabulary(unittest.TestCase):
    def test_cellvit_types_are_global_not_per_roi(self):
        import json as _json

        from pilot.cells import (
            cells_from_cellvit_dir,
            global_type_index,
            load_cells,
        )

        root = Path(__file__).resolve().parent / "_tmp_types"
        shutil.rmtree(root, ignore_errors=True)
        try:
            (root / "json").mkdir(parents=True)
            (root / "json" / "a_cells.json").write_text(
                _json.dumps(
                    {
                        "cells": [
                            {"centroid": [1, 1], "class": "Tumor"},
                            {"centroid": [2, 2], "class": "Stroma"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            # ROI b has only Tumor. A per-ROI factorisation would call it 0
            # here and 1 in ROI a.
            (root / "json" / "b_cells.json").write_text(
                _json.dumps({"cells": [{"centroid": [3, 3], "class": "Tumor"}]}),
                encoding="utf-8",
            )
            cells_from_cellvit_dir(root / "json", root / "cells")
            a_pts, a_types = load_cells(root / "cells" / "a.csv")
            b_pts, b_types = load_cells(root / "cells" / "b.csv")
            indexed, vocab = global_type_index([a_types, b_types])
            tumor = vocab.index("Tumor")
            self.assertEqual(indexed[0][0], tumor)
            self.assertEqual(indexed[1][0], tumor)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestSpatialHeads(unittest.TestCase):
    def test_grid_and_heads(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from pilot.heads import (
            CounterfactualHead,
            MASK_DIM,
            StructureHead,
            grid_from_hidden,
        )

        self.assertEqual(MASK_DIM, 8)
        hidden = torch.randn(1, 9, 4)
        mask = torch.ones(1, 9, dtype=torch.bool)
        grid = grid_from_hidden(hidden, mask, torch.tensor([[1, 6, 6]]), merge_size=2)
        self.assertEqual(tuple(grid.shape), (3, 3, 4))

        struct = StructureHead(4, 5, hidden=8)
        s = struct(grid)
        self.assertEqual(tuple(s.shape), (1, 5))

        cf = CounterfactualHead(4, hidden=8)
        feats = torch.zeros(1, MASK_DIM)
        feats[0, 2] = 0.2
        logit = cf(grid, feats)
        self.assertEqual(tuple(logit.shape), (1,))

    def test_grid_rejects_mismatched_shapes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from pilot.heads import grid_from_hidden

        hidden = torch.randn(1, 8, 4)
        mask = torch.ones(1, 8, dtype=torch.bool)
        self.assertIsNone(grid_from_hidden(hidden, mask, torch.tensor([[1, 6, 6]])))


if __name__ == "__main__":
    unittest.main()
