import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.data import build_cf_examples, make_synthetic, write_jsonl  # noqa: E402


class TestWrongBoxControl(unittest.TestCase):
    def test_end_to_end_on_synthetic(self):
        root = Path(__file__).resolve().parent / "_tmp_wrongbox"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        try:
            df = make_synthetic(root / "data", per_class=2, n_cells=140, size=256, seed=11)
            rows = build_cf_examples(
                df,
                root / "data" / "cells",
                per_roi=4,
                seed=0,
                annotate=True,
                annotate_dir=root / "data" / "annotated",
                min_affected=3,
            )
            self.assertTrue(rows)
            data = root / "cf.jsonl"
            write_jsonl(rows, data)

            out_dir = root / "wrongbox"
            out_jsonl = root / "wrongbox.jsonl"
            script = Path(__file__).resolve().parents[1] / "scripts" / "controls" / "make_wrongbox_control.py"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--data",
                    str(data),
                    "--cells",
                    str(root / "data" / "cells"),
                    "--out",
                    str(out_dir),
                    "--jsonl",
                    str(out_jsonl),
                    "--select-label",
                    "any",
                    "--min-center-shift",
                    "0.10",
                    "--max-iou",
                    "0.50",
                    "--min-affected",
                    "3",
                    "--attempts",
                    "100",
                    "--seed",
                    "0",
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(out_jsonl.exists())
            out = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line.strip()]
            self.assertTrue(out, proc.stdout)
            source_by_id = {r["ex_id"]: r for r in rows}
            for r in out:
                self.assertTrue(r["wrongbox"])
                self.assertIn(r["source_ex_id"], source_by_id)
                self.assertEqual(r["ex_id"], r["source_ex_id"] + "__wrongbox")
                self.assertEqual(r["answer"], "yes" if r["changed"] else "no")
                self.assertTrue(Path(r["image"]).exists())
                self.assertTrue(0.0 <= r["center"][0] <= 1.0)
                self.assertTrue(0.0 <= r["center"][1] <= 1.0)
                self.assertGreater(r["radius"], 0.0)
                # The prompt must stay coordinate-free; only the image changes.
                self.assertNotIn("[", r["question"])
                self.assertEqual(r["question"], source_by_id[r["source_ex_id"]]["question"])
                self.assertEqual(r["descriptor"], source_by_id[r["source_ex_id"]]["descriptor"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_rejects_before_after(self):
        root = Path(__file__).resolve().parent / "_tmp_wrongbox_ba"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        try:
            data = root / "cf.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "ex_id": "x",
                        "roi_id": "x",
                        "presentation": "before_after",
                        "image": "x.png",
                        "image_size": [10, 10],
                        "center": [0.5, 0.5],
                        "radius": 0.1,
                        "changed": 0,
                    }
                )
                + "\n"
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "controls" / "make_wrongbox_control.py"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--data",
                    str(data),
                    "--cells",
                    str(root),
                    "--out",
                    str(root / "out"),
                    "--jsonl",
                    str(root / "out.jsonl"),
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("box", proc.stderr + proc.stdout)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
