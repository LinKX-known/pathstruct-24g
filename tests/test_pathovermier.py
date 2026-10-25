"""ASPECT-metric tests: the numbers in the comparison table must be their numbers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.pathovermier import (  # noqa: E402
    coherent_accuracy,
    count_score,
    rawr,
    summarise,
    within_tolerance,
)


class TestCountScore(unittest.TestCase):
    def test_tolerance_formula(self):
        self.assertTrue(within_tolerance(11, 10))     # tau = max(1, 1) = 1
        self.assertFalse(within_tolerance(12, 10))
        self.assertTrue(within_tolerance(105, 100))   # tau = 10
        self.assertFalse(within_tolerance(111, 100))

    def test_missing_counts_score_zero(self):
        self.assertAlmostEqual(count_score({"a": 10}, {"a": 10, "b": 3}), 0.5)
        self.assertAlmostEqual(count_score({"a": 10, "b": 3}, {"a": 10, "b": 3}), 1.0)
        self.assertAlmostEqual(count_score({}, {"a": 1}), 0.0)

    def test_empty_required_is_nan(self):
        import math

        self.assertTrue(math.isnan(count_score({"a": 1}, {})))


class TestCoherence(unittest.TestCase):
    def test_defined_rows_only(self):
        self.assertAlmostEqual(coherent_accuracy([1, None, 0], [1, 1, 0]), 1.0)
        self.assertAlmostEqual(coherent_accuracy([1, 0], [1, 1]), 0.5)

    def test_all_undefined(self):
        self.assertTrue(coherent_accuracy([None, None], [1, 0]) != coherent_accuracy([None], [1]))


class TestRawr(unittest.TestCase):
    def test_only_correct_and_complete(self):
        # correct+complete rows: index 0 (S=1) and 1 (S=0) -> RAWR = 1 - 0.5
        self.assertAlmostEqual(rawr([1.0, 0.0, 0.2], [True, True, False], [True, True, True]), 0.5)
        self.assertAlmostEqual(rawr([1.0, 1.0], [True, True], [True, False]), 0.0)

    def test_no_eligible_rows(self):
        import math

        self.assertTrue(math.isnan(rawr([1.0], [False], [True])))


class TestSummary(unittest.TestCase):
    def test_table_row(self):
        out = summarise(
            reported=[{"a": 10, "b": 5}, {"a": 3, "b": 5}],
            required=[{"a": 10, "b": 5}, {"a": 10, "b": 5}],
            answer_correct=[True, True],
            decisions=[1, 0],
            expected=[1, 1],
            complete_counts=[True, True],
        )
        self.assertAlmostEqual(out["S"], 0.75)
        self.assertAlmostEqual(out["RAWR"], 0.25)
        self.assertAlmostEqual(out["CA"], 0.5)
        self.assertAlmostEqual(out["answer_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
