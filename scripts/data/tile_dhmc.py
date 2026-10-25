#!/usr/bin/env python
"""Tile DHMC whole-slide images into labelled ROIs.

DHMC (Dartmouth) is the target diagnostic task: 143 lung adenocarcinoma WSIs,
each labelled with its **predominant** growth pattern (lepidic / acinar /
papillary / micropapillary / solid). Request the download from
<https://bmirds.github.io/LungCancer/> -- the links arrive by email and expire
after four hours, so download in batches and tile immediately.

Disk discipline matters: the four zips total ~50 GB and expand to more. Download
one zip, extract, tile, delete, repeat.

    python scripts/data/tile_dhmc.py \
        --wsi-dir raw/dhmc_wsi_4 --meta raw/MetaData.csv \
        --out data/real/dhmc --per-slide 20 --tile 1024

Label caveat, deliberately not hidden: the label is the slide's predominant
pattern, so an ROI inherits it and some ROIs are mislabelled. Train ROI-level,
**evaluate slide-level by majority vote**, and say so in the paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

CLASSES = ("lepidic", "acinar", "papillary", "micropapillary", "solid")


def tissue_fraction(tile: np.ndarray) -> float:
    """Fraction of pixels that look like stained tissue rather than background."""
    rgb = tile.astype(np.float32)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    val = mx / 255.0
    tissue = (sat > 0.12) & (val < 0.92)
    return float(tissue.mean())


def pick_level(slide, target_mpp: float = 0.5) -> int:
    """Level whose resolution is closest to ``target_mpp`` (20x)."""
    base = float(slide.properties.get("openslide.mpp-x", 0.25) or 0.25)
    downsample = max(base / target_mpp, 1.0)
    return int(slide.get_best_level_for_downsample(downsample))


def tiles_from_slide(slide, level: int, tile: int, min_tissue: float, rng, per_slide: int):
    width, height = slide.level_dimensions[level]
    step = tile
    candidates = []
    for y in range(0, max(height - tile, 0), step):
        for x in range(0, max(width - tile, 0), step):
            candidates.append((x, y))
    if not candidates:
        return []
    order = rng.permutation(len(candidates))
    chosen = []
    for idx in order:
        x, y = candidates[int(idx)]
        patch = slide.read_region((x, y), level, (tile, tile)).convert("RGB")
        if tissue_fraction(np.asarray(patch)) < min_tissue:
            continue
        chosen.append((x, y, patch))
        if len(chosen) >= per_slide:
            break
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi-dir", dest="wsi_dir", required=True)
    ap.add_argument("--meta", required=True, help="MetaData.csv from the dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--per-slide", dest="per_slide", type=int, default=20)
    ap.add_argument("--min-tissue", dest="min_tissue", type=float, default=0.45)
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import openslide
    except ImportError:
        print("openslide is required: pip install openslide-python (and the openslide C library)", file=sys.stderr)
        return 2

    meta = pd.read_csv(args.meta)
    name_col = next(c for c in meta.columns if c.lower().startswith("file name"))
    class_col = next(c for c in meta.columns if c.lower() == "class")
    wanted = {c.strip().lower() for c in args.classes.split(",")}
    meta = meta[meta[class_col].str.lower().isin(wanted)]

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    for rec in meta.to_dict(orient="records"):
        stem = str(rec[name_col]).rsplit(".", 1)[0]
        path = Path(args.wsi_dir) / str(rec[name_col])
        if not path.exists():
            print(f"  missing {path.name}, skip")
            continue
        slide = openslide.OpenSlide(str(path))
        level = pick_level(slide)
        tiles = tiles_from_slide(slide, level, args.tile, args.min_tissue, rng, args.per_slide)
        for i, (x, y, patch) in enumerate(tiles):
            roi_id = f"{stem}__{i:03d}"
            dst = img_dir / f"{roi_id}.jpg"
            patch.save(dst, quality=92)
            rows.append(
                {
                    "roi_id": roi_id,
                    "patient_id": stem,
                    "image_path": str(dst),
                    "label": str(rec[class_col]).lower(),
                    "x": x,
                    "y": y,
                }
            )
        print(f"  {stem}: {len(tiles)} ROIs (level {level})", flush=True)
        slide.close()

    if not rows:
        print("no ROIs written", file=sys.stderr)
        return 3
    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    print(f"\n{len(df)} ROIs from {df['patient_id'].nunique()} slides -> {out / 'manifest.csv'}")
    print(dict(df["label"].value_counts()))
    print("\nnext: split by slide, then `pilot build` with --val-frac by patient")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
