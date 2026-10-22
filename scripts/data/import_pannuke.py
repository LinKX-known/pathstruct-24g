#!/usr/bin/env python
"""Convert PanNuke fold1 into the pilot's cell tables and a tissue-held-out split.

PanNuke: real H&E 256x256 patches, expert nucleus masks, 5 cell classes and 19
tissue labels. The patches are small (median 20 nuclei), so they are upscaled
x4 to 1024 and `pilot build` should use ``--min-cells 16``.

The split is by *tissue*, not by patch: four tissues are completely held out.
This is a domain-generalisation test for the structural heads (it also avoids
the WSI-patch leakage that a random patch split would have).

    python scripts/import_pannuke.py \
        --parquet .../fold1-*.parquet --out data/real/pannuke \
        --val-tissues breast,colon,bile,cervix
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pilot.cells import write_cells  # noqa: E402

TISSUE_CODES = [
    "adrenal", "bile", "bladder", "breast", "cervix", "colon", "esophagus",
    "headneck", "kidney", "liver", "lung", "ovarian", "pancreatic", "prostate",
    "skin", "stomach", "testis", "thyroid", "uterus",
]
CELL_TYPES = ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]


def decode(struct) -> Image.Image:
    return Image.open(io.BytesIO(struct["bytes"]))


def process(
    parquet: str,
    out: Path,
    val_tissues: set[str],
    per_train: int,
    per_val: int,
    min_cells: int,
    upscale: int,
):
    img_dir = out / "images"
    cell_dir = out / "cells"
    img_dir.mkdir(parents=True, exist_ok=True)
    cell_dir.mkdir(parents=True, exist_ok=True)

    counts = {t: 0 for t in TISSUE_CODES}
    train_rows, val_rows = [], []
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(
        batch_size=64, columns=["image", "instances", "categories", "tissue"]
    ):
        images = batch["image"].to_pylist()
        instances = batch["instances"].to_pylist()
        categories = batch["categories"].to_pylist()
        tissues = batch["tissue"].to_pylist()
        for image_struct, mask_structs, cats, tissue in zip(
            images, instances, categories, tissues
        ):
            tissue_code = TISSUE_CODES[int(tissue)]
            split = "val" if tissue_code in val_tissues else "train"
            limit = per_val if split == "val" else per_train
            if counts[tissue_code] >= limit:
                continue
            if len(mask_structs) < min_cells:
                continue

            roi_id = f"pannuke_{split}_{tissue_code}_{counts[tissue_code]:03d}"
            image = decode(image_struct).convert("RGB")
            image = image.resize(
                (image.size[0] * upscale, image.size[1] * upscale), Image.BICUBIC
            )
            image.save(img_dir / f"{roi_id}.png")

            pts, types = [], []
            for mask_struct, cat in zip(mask_structs, cats):
                arr = np.asarray(decode(mask_struct))
                ys, xs = np.nonzero(arr)
                if len(xs) < 4:
                    continue
                pts.append((float(xs.mean()) * upscale, float(ys.mean()) * upscale))
                types.append(CELL_TYPES[min(int(cat), len(CELL_TYPES) - 1)])
            if len(pts) < min_cells:
                continue
            write_cells(
                np.asarray(pts, dtype=np.float64),
                np.asarray(types, dtype=object),
                cell_dir / f"{roi_id}.csv",
            )
            row = {
                "roi_id": roi_id,
                "patient_id": f"tissue_{tissue_code}",
                "image_path": str(img_dir / f"{roi_id}.png"),
                "label": tissue_code,
            }
            (val_rows if split == "val" else train_rows).append(row)
            counts[tissue_code] += 1
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default="data/real/pannuke")
    ap.add_argument("--val-tissues", default="breast,colon,bile,cervix")
    ap.add_argument("--per-train", dest="per_train", type=int, default=30)
    ap.add_argument("--per-val", dest="per_val", type=int, default=40)
    ap.add_argument("--min-cells", dest="min_cells", type=int, default=16)
    ap.add_argument("--upscale", type=int, default=4)
    args = ap.parse_args()

    val_tissues = {t.strip() for t in args.val_tissues.split(",") if t.strip()}
    unknown = val_tissues - set(TISSUE_CODES)
    if unknown:
        raise SystemExit(f"unknown val tissues: {sorted(unknown)}")

    out = Path(args.out)
    train, val, counts = process(
        args.parquet,
        out,
        val_tissues,
        args.per_train,
        args.per_val,
        args.min_cells,
        args.upscale,
    )
    train.to_csv(out / "manifest_train.csv", index=False)
    val.to_csv(out / "manifest_test.csv", index=False)
    print("tissue counts:", {k: v for k, v in counts.items() if v})
    print(f"train {len(train)} patches | held-out {len(val)} patches")
    print(f"manifests written under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
