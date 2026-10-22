#!/usr/bin/env python
"""Convert the RationAI/MoNuSeg parquet exports into the pilot's cell tables.

The HF export stores each image as PNG bytes and each nucleus as a binary PNG
mask. We read it with pyarrow (no `datasets` dependency; the latest datasets
requires huggingface-hub>=1.31, which conflicts with transformers 4.57).

Outputs:
    data/real/monuseg/images/{roi_id}.png
    data/real/monuseg/cells/{roi_id}.csv          (x, y, type)
    data/real/monuseg/manifest_train.csv
    data/real/monuseg/manifest_test.csv

The provided train/test split is patient-disjoint, so the test split is a
clean external validation. Cell types are not annotated in MoNuSeg; every
nucleus gets type ``nucleus``.
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

TISSUE = {
    0: "unknown",
    1: "breast",
    2: "kidney",
    3: "liver",
    4: "prostate",
    5: "bladder",
    6: "colon",
    7: "stomach",
}


def decode_image(struct) -> Image.Image:
    data = struct["bytes"]
    return Image.open(io.BytesIO(data)).convert("RGB")


def decode_mask(struct) -> np.ndarray:
    data = struct["bytes"]
    return np.asarray(Image.open(io.BytesIO(data)))


def process_split(parquet: str, split: str, out: Path) -> pd.DataFrame:
    img_dir = out / "images"
    cell_dir = out / "cells"
    img_dir.mkdir(parents=True, exist_ok=True)
    cell_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(
        batch_size=8, columns=["patient", "image", "instances", "tissue"]
    ):
        patients = batch["patient"].to_pylist()
        images = batch["image"].to_pylist()
        instances = batch["instances"].to_pylist()
        tissues = batch["tissue"].to_pylist()
        for patient, image_struct, mask_structs, tissue in zip(
            patients, images, instances, tissues
        ):
            roi_id = f"monuseg_{split}_{len(rows):03d}"
            image = decode_image(image_struct)
            w, h = image.size
            image.save(img_dir / f"{roi_id}.png")

            pts, types = [], []
            for mask_struct in mask_structs:
                arr = decode_mask(mask_struct)
                ys, xs = np.nonzero(arr)
                if len(xs) < 10:
                    continue
                pts.append((float(xs.mean()), float(ys.mean())))
                types.append("nucleus")
            if not pts:
                continue
            write_cells(
                np.asarray(pts, dtype=np.float64),
                np.asarray(types, dtype=object),
                cell_dir / f"{roi_id}.csv",
            )
            label = TISSUE.get(int(tissue), "unknown")
            rows.append(
                {
                    "roi_id": roi_id,
                    "patient_id": str(patient),
                    "image_path": str(img_dir / f"{roi_id}.png"),
                    "label": label,
                }
            )
            if len(rows) % 10 == 0:
                print(f"  {split}: {len(rows)} images", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--out", default="data/real/monuseg")
    args = ap.parse_args()

    out = Path(args.out)
    train = process_split(args.train_parquet, "train", out)
    test = process_split(args.test_parquet, "test", out)
    train.to_csv(out / "manifest_train.csv", index=False)
    test.to_csv(out / "manifest_test.csv", index=False)
    print(
        f"train {len(train)} images / {train['patient_id'].nunique()} patients | "
        f"test {len(test)} images / {test['patient_id'].nunique()} patients"
    )
    print(f"manifests written under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
