"""The cohort checker must pass a clean layout and fail a broken one."""

import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "check_dataset.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_dataset", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCheckDataset(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / "_tmp_check"
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "images").mkdir(parents=True)
        (self.root / "cells").mkdir(parents=True)
        rows = []
        for i, label in enumerate(["ring", "solid", "ring"]):
            roi = f"roi_{i}"
            img = self.root / "images" / f"{roi}.png"
            Image.new("RGB", (64, 64), (200, 150, 180)).save(img)
            n = 80 if i != 2 else 10
            pd.DataFrame({"x": range(n), "y": range(n), "type": ["cell"] * n}).to_csv(
                self.root / "cells" / f"{roi}.csv", index=False
            )
            rows.append(
                {"roi_id": roi, "patient_id": f"p{i}", "image_path": str(img), "label": label}
            )
        self.manifest = self.root / "manifest.csv"
        pd.DataFrame(rows).to_csv(self.manifest, index=False)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def run_checker(self, *extra):
        mod = load_checker()
        sys.argv = [
            "check_dataset",
            "--manifest", str(self.manifest),
            "--cells", str(self.root / "cells"),
            *extra,
        ]
        return mod.main()

    def test_clean_layout_passes(self):
        # roi_2 has 10 cells on purpose; the clean check uses a threshold below it
        self.assertEqual(self.run_checker("--min-cells", "5"), 0)

    def test_missing_image_fails(self):
        (self.root / "images" / "roi_1.png").unlink()
        self.assertEqual(self.run_checker("--min-cells", "60"), 1)

    def test_too_few_cells_fails(self):
        self.assertEqual(self.run_checker("--min-cells", "60"), 1)

    def test_split_leak_fails(self):
        val = self.root / "val.csv"
        pd.DataFrame(
            [{"roi_id": "roi_9", "patient_id": "p0", "image_path": "x", "label": "ring"}]
        ).to_csv(val, index=False)
        self.assertEqual(
            self.run_checker("--min-cells", "60", "--val-manifest", str(val)), 1
        )


if __name__ == "__main__":
    unittest.main()
