#!/usr/bin/env python
"""Build the box-only control: same annotation, no cells.

This is the strongest cheap check that a counterfactual result uses the cell
arrangement rather than the box position, the arrow, or the edit kind. It
draws exactly the green box / orange arrow on a blank canvas and rewrites the
``image`` field, so the prompt is unchanged.

    python scripts/make_blank_control.py \
        --data data/synth/cf_val.jsonl --out data/synth/blank \
        --jsonl data/synth/cf_val_blank.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.data import (  # noqa: E402
    annotate_edit,
    read_jsonl,
    render_cells,
    side_by_side,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="directory for blank annotated images")
    ap.add_argument("--jsonl", required=True, help="output JSONL with rewritten image paths")
    ap.add_argument("--background", default="247,244,240", help="R,G,B background")
    ap.add_argument("--presentation", choices=("box", "before_after"), default="box")
    ap.add_argument("--dot-radius", dest="dot_radius", type=float, default=8.0)
    args = ap.parse_args()

    rgb = tuple(int(v) for v in args.background.split(","))
    if len(rgb) != 3:
        raise SystemExit("--background must be R,G,B")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.data)
    result = []
    for r in rows:
        dst = out_dir / f"{r['ex_id']}.png"
        if args.presentation == "before_after":
            panel_w, panel_h = r.get("panel_size", r["image_size"])
            cw, ch = r["image_size"]
            if "panel_size" not in r:
                panel_w, panel_h = (cw - 4) // 2, ch
            cx, cy = r["center"]
            rad = r["radius"] * max(panel_w, panel_h)
            box = (
                max(0.0, cx * panel_w - rad),
                max(0.0, cy * panel_h - rad),
                min(panel_w - 1.0, cx * panel_w + rad),
                min(panel_h - 1.0, cy * panel_h + rad),
            )
            left = render_cells(
                np.zeros((0, 2)),
                None,
                (panel_w, panel_h),
                dot_radius=args.dot_radius,
                box=box,
                label="before",
            )
            right = render_cells(
                np.zeros((0, 2)),
                None,
                (panel_w, panel_h),
                dot_radius=args.dot_radius,
                box=box,
                label="after",
            )
            side_by_side(left, right).save(dst)
        else:
            w, h = r["image_size"]
            blank = out_dir / "_blank.png"
            Image.new("RGB", (w, h), rgb).save(blank)
            cx, cy = r["center"]
            vx, vy = r.get("vector", [0.0, 0.0])
            annotate_edit(
                blank,
                dst,
                (cx * w, cy * h),
                r["radius"] * max(w, h),
                (vx * w, vy * h),
                r["kind"],
            )
        q = dict(r)
        q["image"] = str(dst)
        q["source_image"] = str(dst)
        result.append(q)
    write_jsonl(result, args.jsonl)
    print(f"wrote {len(result)} blank examples to {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
