import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.data import build_growth_examples, make_synthetic, read_manifest  # noqa: E402


class TestGrowthDescriptors(unittest.TestCase):
    def test_descriptor_attachment(self):
        root = Path(__file__).resolve().parent / "_tmp_growth_desc"
        shutil.rmtree(root, ignore_errors=True)
        try:
            make_synthetic(root, per_class=1, n_cells=80, size=128, seed=4)
            manifest = read_manifest(root / "manifest.csv")
            plain = build_growth_examples(manifest)
            with_desc = build_growth_examples(
                manifest, cells_dir=root / "cells", with_descriptor=True
            )
            self.assertEqual(len(plain), len(with_desc))
            self.assertTrue(all("descriptor" not in r for r in plain))
            self.assertTrue(all("descriptor" in r for r in with_desc))
            for r in with_desc:
                self.assertGreater(len(r["descriptor"]), 0)
                self.assertEqual(len(r["descriptor"]), len(r["descriptor_names"]))
                self.assertTrue(np.all(np.isfinite(r["descriptor"])))
                self.assertGreater(r["n_cells"], 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
