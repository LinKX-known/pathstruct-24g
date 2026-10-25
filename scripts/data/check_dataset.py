#!/usr/bin/env python
"""Check a prepared cohort before spending GPU time on it.

    python scripts/data/check_dataset.py --manifest data/real/pannuke/manifest.csv \
        --cells data/real/pannuke/cells

Exits 0 only when the cohort is safe to train on. Every check prints what it found
rather than raising, so one run tells you everything that is wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("roi_id", "patient_id", "image_path", "label")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--val-manifest", dest="val_manifest", default=None)
    ap.add_argument("--min-cells", dest="min_cells", type=int, default=60)
    args = ap.parse_args()

    problems: list[str] = []
    df = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        print(f"FAIL missing columns: {missing_cols}")
        return 1

    df["image_ok"] = df["image_path"].map(lambda p: bool(p) and Path(p).exists())
    df["cells_ok"] = df["roi_id"].map(lambda r: (Path(args.cells) / f"{r}.csv").exists())
    n_img = int((~df["image_ok"]).sum())
    n_cells = int((~df["cells_ok"]).sum())
    if n_img:
        problems.append(f"{n_img} ROIs have a missing image")
    if n_cells:
        problems.append(f"{n_cells} ROIs have no cells/{{roi_id}}.csv")

    ok = df[df["image_ok"] & df["cells_ok"]]
    counts = []
    for roi in ok["roi_id"]:
        try:
            counts.append(len(pd.read_csv(Path(args.cells) / f"{roi}.csv")))
        except Exception:
            counts.append(0)
    counts_arr = np.asarray(counts) if counts else np.asarray([0])
    too_few = int((counts_arr < args.min_cells).sum())
    if too_few:
        problems.append(
            f"{too_few} ROIs below --min-cells {args.min_cells} "
            f"(median {np.median(counts_arr):.0f}, min {counts_arr.min()})"
        )

    labels = ok["label"].value_counts()
    print("label counts:", dict(labels))
    if len(labels) < 2:
        problems.append("fewer than two classes")
    elif labels.iloc[0] > 0.85 * len(ok):
        problems.append(f"class imbalance: {labels.iloc[0]}/{len(ok)} are '{labels.index[0]}'")

    if args.val_manifest:
        val = pd.read_csv(args.val_manifest, dtype=str, keep_default_na=False)
        shared = set(ok["patient_id"]) & set(val.get("patient_id", []))
        if shared:
            problems.append(f"{len(shared)} patients appear in both train and val, e.g. {list(shared)[:3]}")
        print(f"val rows {len(val)} | shared patients {len(shared)}")

    print(f"rows {len(df)} | with image+cells {len(ok)} | "
          f"cells median {np.median(counts_arr):.0f} min {counts_arr.min()} max {counts_arr.max()}")

    if problems:
        print("\nFAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("\nOK: safe to train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
