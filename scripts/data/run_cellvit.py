#!/usr/bin/env python
"""Run the official CellViT-Inference pipeline, then convert its output.

We do not reimplement segmentation. The supported package is ``cellvit``
(``pip install cellvit``), whose CLI is config-driven:

    cellvit-inference --config path/to/config.yaml

This wrapper exists for two reasons only:

1. it calls that binary instead of importing research code;
2. it converts ``*_cells.json`` into the ``cells/{roi_id}.csv`` layout the rest of
   this repository consumes (the converter is :func:`pilot.cells.cells_from_cellvit_dir`).

    python scripts/data/run_cellvit.py \
        --config cellvit_config.yaml \
        --json-dir out/cellvit/monuseg \
        --cells-out data/real/monuseg/cells

If ``--json-dir`` is not the directory the config writes to, the conversion reports
zero files -- check the config's output path.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pilot.cells import cells_from_cellvit_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="CellViT-Inference YAML config")
    ap.add_argument("--json-dir", dest="json_dir", required=True, help="where the config writes *_cells.json")
    ap.add_argument("--cells-out", dest="cells_out", required=True)
    ap.add_argument("--skip-run", dest="skip_run", action="store_true", help="only convert existing output")
    ap.add_argument("--suffix", default="_cells")
    args = ap.parse_args()

    if not args.skip_run:
        binary = shutil.which("cellvit-inference")
        if binary is None:
            print(
                "cellvit-inference not found. Install the official package:\n"
                "  pip install cellvit\n"
                "and see https://tio-ikim.github.io/CellViT-Inference/",
                file=sys.stderr,
            )
            return 2
        cmd = [binary, "--config", args.config]
        print("running:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"cellvit-inference failed with code {proc.returncode}", file=sys.stderr)
            return proc.returncode

    df = cells_from_cellvit_dir(args.json_dir, args.cells_out, suffix=args.suffix)
    if df.empty:
        print(
            f"no *{args.suffix}.json found in {args.json_dir}; "
            "check the output path in the CellViT config",
            file=sys.stderr,
        )
        return 3
    print(f"converted {len(df)} ROIs | cells median {df['n_cells'].median():.0f}")
    bad = df[df["error"] != ""]
    if len(bad):
        print(bad.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
