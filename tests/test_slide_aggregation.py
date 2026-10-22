import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "eval"
    / "aggregate_slide_predictions.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("aggregate_slide_predictions", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSlideAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_majority_vote_and_ties(self):
        rows = [
            {"wsi_id": "s1", "gold": "acinar", "pred": "acinar"},
            {"wsi_id": "s1", "gold": "acinar", "pred": "acinar"},
            {"wsi_id": "s1", "gold": "acinar", "pred": "solid"},
            {"wsi_id": "s2", "gold": "solid", "pred": "solid"},
            {"wsi_id": "s2", "gold": "solid", "pred": "solid"},
            {"wsi_id": "s2", "gold": "solid", "pred": ""},
            {"wsi_id": "s3", "gold": "lepidic", "pred": "acinar"},
            {"wsi_id": "s3", "gold": "lepidic", "pred": "solid"},
        ]
        slides = self.mod.aggregate(rows)
        by_id = {s["slide"]: s for s in slides}
        self.assertEqual(by_id["s1"]["pred"], "acinar")
        self.assertFalse(by_id["s1"]["tie"])
        self.assertEqual(by_id["s2"]["pred"], "solid")
        self.assertEqual(by_id["s2"]["parse_failures"], 1)
        self.assertTrue(by_id["s3"]["tie"])
        # alphabetical tie-break is deterministic
        self.assertEqual(by_id["s3"]["pred"], "acinar")
        score = self.mod.score(slides)
        self.assertEqual(score["n_slides"], 3)
        self.assertAlmostEqual(score["accuracy"], 2 / 3)
        self.assertEqual(score["ties"], 1)

    def test_cli_roundtrip(self):
        rows = [
            {"wsi_id": "s1", "gold": "acinar", "pred": "acinar"},
            {"wsi_id": "s1", "gold": "acinar", "pred": "solid"},
            {"wsi_id": "s2", "gold": "solid", "pred": "solid"},
            {"wsi_id": "s2", "gold": "solid", "pred": "solid"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "pred.json"
            out = root / "out.json"
            pred.write_text(json.dumps(rows), encoding="utf-8")
            proc = __import__("subprocess").run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--pred",
                    str(pred),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(out.read_text())
            self.assertEqual(payload["score"]["n_slides"], 2)
            self.assertEqual(payload["score"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
