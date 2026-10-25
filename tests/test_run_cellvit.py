"""The CellViT runner must insist on CellViT++ and find the output directory."""

import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "run_cellvit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_cellvit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCellVitConfig(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / "_tmp_cellvit_cfg"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, text: str) -> str:
        path = self.root / "cfg.yaml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_sam_is_accepted_and_outdir_returned(self):
        mod = load_module()
        cfg = self.write(
            "model: SAM\noutput_format:\n  outdir: out/cellvit\n"
            "process_dataset:\n  wsi_folder: data/x\n"
        )
        code, outdir = mod.check_config(cfg)
        self.assertEqual(code, 0)
        self.assertEqual(outdir, "out/cellvit")

    def test_hipt_is_accepted(self):
        mod = load_module()
        code, _ = mod.check_config(self.write("model: HIPT\noutput_format:\n  outdir: o\n"))
        self.assertEqual(code, 0)

    def test_basic_cellvit_is_rejected(self):
        mod = load_module()
        code, _ = mod.check_config(self.write("model: ViT256\noutput_format:\n  outdir: o\n"))
        self.assertNotEqual(code, 0)

    def test_template_config_is_valid(self):
        mod = load_module()
        code, outdir = mod.check_config(str(ROOT / "configs" / "cellvit_sam_h.yaml"))
        self.assertEqual(code, 0)
        self.assertTrue(outdir)


if __name__ == "__main__":
    unittest.main()
