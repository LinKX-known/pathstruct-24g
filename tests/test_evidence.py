"""Tests for the evidence format: staging, masking, parsing, scoring."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.evidence import (  # noqa: E402
    evidence_index,
    mask_prompt,
    parse_generated,
    score_evidence,
    target_for_stage,
    think_block,
)
from pilot.tokens import CELL, STRUCT, UNI, build_response, bin_descriptor, observe_sentence  # noqa: E402

ROW = {
    "question": "What is the predominant growth pattern?",
    "answer": "acinar",
    "observe": {"components": "1", "lumina": "3+", "lumen_size": "large"},
    "descriptor": [1.0, 3.0, 0.3, 2.0, 9.0],
}


class TestStages(unittest.TestCase):
    def test_perceive_puts_tokens_in_prompt(self):
        prefix, target = target_for_stage(ROW, "perceive")
        self.assertEqual(prefix, think_block())
        self.assertIn("<|observe|>", target)
        self.assertNotIn("<|think|>", target)

    def test_generate_and_reason_include_tokens(self):
        for stage in ("generate", "reason"):
            prefix, target = target_for_stage(ROW, stage)
            self.assertEqual(prefix, "")
            self.assertIn("<|think|>", target)
            self.assertIn("<|struct|>", target)

    def test_unknown_stage(self):
        with self.assertRaises(ValueError):
            target_for_stage(ROW, "nope")


class TestMaskingAndPositions(unittest.TestCase):
    def test_mask_prompt(self):
        labels = mask_prompt([5, 6, 7, 8], 2)
        self.assertEqual(labels, [-100, -100, 7, 8])

    def test_evidence_index(self):
        ids = {UNI: 10, CELL: 11, STRUCT: 12}
        seq = [9, 10, 10, 11, 11, 11, 12, 12]
        pos = evidence_index(seq, ids)
        self.assertEqual(pos["uni"], [1, 2])
        self.assertEqual(pos["cell"], [3, 4, 5])
        self.assertEqual(pos["struct"], [6, 7])


class TestRoundTrip(unittest.TestCase):
    def test_bins_match_parsing(self):
        bins = bin_descriptor([1.0, 3.0, 0.3, 2.0, 9.0], ["n_comp", "hole_count", "hole_area_frac", "h0_max", "h0_sum"])
        self.assertEqual(bins, {"components": "1", "lumina": "3+", "lumen_size": "large"})
        text = build_response(observe_sentence(bins), bins)
        parsed = parse_generated(text)
        self.assertEqual(parsed["observe"], bins)
        self.assertEqual(parsed["answer"], observe_sentence(bins))

    def test_parse_missing_blocks(self):
        parsed = parse_generated("no structure at all")
        self.assertIsNone(parsed["observe"])
        self.assertEqual(parsed["answer"], "")


class TestScoring(unittest.TestCase):
    def test_metrics(self):
        rows = [
            {
                "gold_answer": "acinar",
                "gold_observe": {"components": "1", "lumina": "3+"},
                "answer": "acinar",
                "observe": {"components": "1", "lumina": "3+"},
            },
            {
                "gold_answer": "solid",
                "gold_observe": {"components": "1", "lumina": "0"},
                "answer": "solid",
                "observe": {"components": "1", "lumina": "1"},
            },
            {
                "gold_answer": "solid",
                "gold_observe": {"components": "1", "lumina": "0"},
                "answer": "acinar",
                "observe": {"components": "1", "lumina": "0"},
            },
        ]
        m = score_evidence(rows)
        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["answer_accuracy"], 2 / 3)
        self.assertAlmostEqual(m["observation_field_accuracy"], 5 / 6)
        self.assertAlmostEqual(m["observation_exact_match"], 2 / 3)
        self.assertAlmostEqual(m["rawr_struct"], 0.5)


if __name__ == "__main__":
    unittest.main()
