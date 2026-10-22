import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.structure_eval import (  # noqa: E402
    format_table,
    log1p_target,
    per_dim_diagnostics,
    summary,
)


class TestStructureEval(unittest.TestCase):
    def test_log1p_target_clips_and_transforms(self):
        x = np.array([[-1.0, 0.0, 3.0], [4.0, 5.0, 6.0]])
        out = log1p_target(x)
        self.assertAlmostEqual(out[0, 0], 0.0)
        self.assertAlmostEqual(out[0, 1], 0.0)
        self.assertAlmostEqual(out[0, 2], np.log1p(3.0))

    def test_perfect_prediction(self):
        true = np.array([[0.0], [1.0], [2.0], [3.0]])
        diags = per_dim_diagnostics(true, true.copy(), train_mean=true.mean(0))
        d = diags[0]
        self.assertAlmostEqual(d["r2"], 1.0)
        self.assertAlmostEqual(d["mae"], 0.0)
        self.assertAlmostEqual(d["spearman"], 1.0)
        self.assertAlmostEqual(d["skill_vs_train_mean"], 1.0)

    def test_zero_inflated_dimension_is_flagged(self):
        true = np.array([[0.0], [0.0], [0.0], [0.0], [1.0]])
        pred = np.full_like(true, 0.5)
        diags = per_dim_diagnostics(true, pred, train_mean=true.mean(0))
        d = diags[0]
        self.assertTrue(d["degenerate"])
        self.assertEqual(d["degen_reason"], "zero-inflated")
        self.assertLess(d["r2"], -0.5)
        self.assertTrue(np.isfinite(d["presence_auc"]))

    def test_train_mean_baseline_and_skill(self):
        true = np.array([[0.0], [1.0], [2.0], [3.0]])
        pred = np.array([[0.2], [0.9], [2.1], [2.8]])
        train_mean = np.array([1.5])
        d = per_dim_diagnostics(true, pred, train_mean=train_mean)[0]
        self.assertTrue(np.isfinite(d["r2_train_mean"]))
        self.assertTrue(np.isfinite(d["skill_vs_train_mean"]))
        self.assertLess(d["mae"], d["mae_train_mean"])

    def test_summary_excludes_degenerate_from_headline(self):
        true = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 1.0],
            ]
        )
        pred = true.copy()
        pred[:, 1] = 0.5
        diags = per_dim_diagnostics(true, pred, names=["good", "sparse"])
        s = summary(diags)
        self.assertEqual(s["n_dims"], 2)
        self.assertEqual(s["n_degenerate"], 1)
        self.assertAlmostEqual(s["r2_mean_nondegenerate"], 1.0)
        self.assertLess(s["r2_mean_all"], 0.5)  # the sparse dim drags the naive mean down
        # The table must not raise and must include the degenerate flag.
        table = format_table(diags)
        self.assertIn("sparse", table)

    def test_constant_dimension_has_nan_r2(self):
        true = np.zeros((4, 1))
        diags = per_dim_diagnostics(true, true.copy())
        self.assertTrue(diags[0]["degenerate"])
        self.assertTrue(np.isnan(diags[0]["r2"]))


if __name__ == "__main__":
    unittest.main()
