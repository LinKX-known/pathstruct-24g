#!/usr/bin/env python
"""Import the Dartmouth (DHMC) lung adenocarcinoma growth-pattern dataset.

The public release is 143 H&E WSIs with a **slide-level predominant growth
pattern** (lepidic / acinar / papillary / micropapillary / solid).  It is not a
patch-level dataset: the standard protocol is to sample patches per slide,
train a patch classifier, and aggregate patch predictions to a slide answer
(or use MIL).  This script does the sampling and writes a manifest that the
rest of the pipeline can consume.

Data access
-----------
The WSIs are available from https://bmirds.github.io/LungCancer/ after a
request form (download links expire).  Point ``--wsi-dir`` at the extracted
``.tif`` files and ``--metadata`` at the released ``MetaData.csv``.  The
detailed label CSV distributed with dependency-mil
(``DETAILED_DHMC_LUAD_LUSC_BENIGN.csv``) is also supported; its
``LUAD_<pattern>`` columns use ``2`` for the predominant pattern.

    python scripts/data/import_dhmc.py \
        --metadata MetaData.csv --wsi-dir /data/DHMC --out data/real/dhmc \
        --patch-size 512 --patches-per-slide 32

``openslide-python`` is strongly recommended for the pyramidal TIFFs
(``pip install 'pathstruct-24g[wsi]'``).  A PIL fallback exists for small test
images but loads the whole file into memory.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LABELS = ("lepidic", "acinar", "papillary", "micropapillary", "solid")
_SYNONYMS = {
    "lepidic": "lepidic",
    "lepidic predominant": "lepidic",
    "lepidic-predominant": "lepidic",
    "acinar": "acinar",
    "acinar predominant": "acinar",
    "papillary": "papillary",
    "papillary predominant": "papillary",
    "micropapillary": "micropapillary",
    "micro-papillary": "micropapillary",
    "micropapillary predominant": "micropapillary",
    "solid": "solid",
    "solid predominant": "solid",
}


def normalise_label(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if text in _SYNONYMS:
        return _SYNONYMS[text]
    for key, label in _SYNONYMS.items():
        if text.startswith(key):
            return label
    return None


def _find_col(df: pd.DataFrame, names) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in lower:
            return lower[name]
    return None


def parse_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the many DHMC metadata variants to wsi_id/patient_id/label."""
    slide_col = _find_col(
        df, ("wsi_id", "slide_id", "slide", "file", "filename", "image", "image_id", "name")
    )
    if slide_col is None:
        raise ValueError(f"cannot find a slide-id column in {list(df.columns)}")
    patient_col = _find_col(df, ("patient_id", "patient", "case_id", "case"))
    label_col = _find_col(
        df, ("label", "pattern", "growth_pattern", "growth pattern", "class", "subtype", "diagnosis")
    )
    detailed = [c for c in df.columns if str(c).lower().startswith("luad_")]

    labels: List[Optional[str]] = []
    for _, row in df.iterrows():
        label = normalise_label(row[label_col]) if label_col is not None else None
        if label is None and detailed:
            values = np.asarray([pd.to_numeric(row[c], errors="coerce") for c in detailed], dtype=float)
            idx = np.flatnonzero(values == 2)
            if len(idx) == 0:
                idx = np.flatnonzero(values == 1)
            if len(idx) == 1:
                label = normalise_label(detailed[int(idx[0])][len("LUAD_"):])
        labels.append(label)

    out = pd.DataFrame(
        {
            "wsi_id": [Path(str(v)).stem for v in df[slide_col].astype(str)],
            "patient_id": (
                df[patient_col].astype(str)
                if patient_col is not None
                else [Path(str(v)).stem for v in df[slide_col].astype(str)]
            ),
            "label": labels,
        }
    )
    out = out[out["label"].notna()].drop_duplicates("wsi_id").reset_index(drop=True)
    return out


class WSReader:
    """Read regions from a pyramidal WSI, with a PIL fallback for small files."""

    def __init__(self, path: Path):
        self.path = path
        self.kind = "pil"
        self._slide = None
        try:
            import openslide

            self._slide = openslide.OpenSlide(str(path))
            self.kind = "openslide"
        except Exception:
            from PIL import Image

            self._pil = Image.open(path).convert("RGB")
        if self.kind == "openslide":
            self.level_count = int(self._slide.level_count)
            self.level_dimensions = [tuple(map(int, d)) for d in self._slide.level_dimensions]
            self.dimensions = tuple(map(int, self._slide.dimensions))
        else:
            self.level_count = 1
            self.level_dimensions = [self._pil.size]
            self.dimensions = self._pil.size

    def read_region(self, x: int, y: int, level: int, size: int):
        from PIL import Image

        if self.kind == "openslide":
            return self._slide.read_region((int(x), int(y)), int(level), (size, size)).convert("RGB")
        # PIL fallback: coordinates are level-0 pixels; only level 0 is supported
        if level != 0:
            raise ValueError("the PIL fallback supports level 0 only")
        return self._pil.crop((int(x), int(y), int(x) + size, int(y) + size))

    def thumbnail(self, max_size: int = 1024):
        from PIL import Image

        if self.kind == "openslide":
            return self._slide.get_thumbnail((max_size, max_size)).convert("RGB")
        im = self._pil.copy()
        im.thumbnail((max_size, max_size), Image.BILINEAR)
        return im

    def close(self):
        if self.kind == "openslide":
            self._slide.close()
        else:
            self._pil.close()


def tissue_mask(thumb) -> np.ndarray:
    from PIL import Image
    from skimage.filters import threshold_otsu

    gray = np.asarray(thumb.convert("L"), dtype=np.float64) / 255.0
    try:
        threshold = threshold_otsu(gray)
    except ValueError:
        return np.zeros(gray.shape, dtype=bool)
    # H&E background is bright; tissue is darker than the Otsu threshold.
    return gray < threshold


def candidate_patches(width, height, patch, stride):
    xs = list(range(0, max(width - patch + 1, 1), max(stride, 1)))
    ys = list(range(0, max(height - patch + 1, 1), max(stride, 1)))
    return [(x, y) for y in ys for x in xs]


def import_slides(
    meta: pd.DataFrame,
    wsi_dir: Path,
    out_dir: Path,
    patch_size: int,
    stride: int,
    level: int,
    patches_per_slide: int,
    min_tissue_frac: float,
    thumb_size: int,
    seed: int,
    max_slides: Optional[int] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    slide_rows: List[Dict[str, Any]] = []

    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    files = {
        p.stem.lower(): p
        for p in wsi_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    }
    for n, rec in enumerate(meta.itertuples(index=False)):
        if max_slides and n >= max_slides:
            break
        path = files.get(str(rec.wsi_id).lower())
        if path is None:
            slide_rows.append(
                {"wsi_id": rec.wsi_id, "patient_id": rec.patient_id, "label": rec.label,
                 "n_patches": 0, "error": "wsi not found"}
            )
            continue
        try:
            reader = WSReader(path)
        except Exception as exc:  # noqa: BLE001
            slide_rows.append(
                {"wsi_id": rec.wsi_id, "patient_id": rec.patient_id, "label": rec.label,
                 "n_patches": 0, "error": str(exc)[:160]}
            )
            continue

        try:
            lvl = min(max(level, 0), reader.level_count - 1)
            width, height = reader.level_dimensions[lvl]
            if width < patch_size or height < patch_size:
                slide_rows.append(
                    {"wsi_id": rec.wsi_id, "patient_id": rec.patient_id, "label": rec.label,
                     "n_patches": 0, "error": "image smaller than patch"}
                )
                continue
            thumb = reader.thumbnail(thumb_size)
            mask = tissue_mask(thumb)
            sx = thumb.size[0] / max(width, 1)
            sy = thumb.size[1] / max(height, 1)
            candidates = candidate_patches(width, height, patch_size, stride)
            rng.shuffle(candidates)
            chosen = []
            for x, y in candidates:
                x0, y0 = int(x * sx), int(y * sy)
                x1, y1 = int((x + patch_size) * sx), int((y + patch_size) * sy)
                region = mask[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
                if region.size == 0 or region.mean() < min_tissue_frac:
                    continue
                chosen.append((x, y, float(region.mean())))
                if len(chosen) >= patches_per_slide:
                    break

            for i, (x, y, frac) in enumerate(chosen):
                roi_id = f"{rec.wsi_id}_p{i:05d}"
                dst = image_dir / f"{roi_id}.png"
                patch = reader.read_region(x, y, lvl, patch_size)
                patch.save(dst)
                rows.append(
                    {
                        "roi_id": roi_id,
                        "patient_id": rec.patient_id,
                        "image_path": str(dst),
                        "label": rec.label,
                        "wsi_id": rec.wsi_id,
                        "x": int(x),
                        "y": int(y),
                        "level": int(lvl),
                        "tissue_frac": round(frac, 4),
                    }
                )
            slide_rows.append(
                {"wsi_id": rec.wsi_id, "patient_id": rec.patient_id, "label": rec.label,
                 "n_patches": len(chosen), "error": ""}
            )
        finally:
            reader.close()

    manifest_cols = [
        "roi_id", "patient_id", "image_path", "label", "wsi_id",
        "x", "y", "level", "tissue_frac",
    ]
    slide_cols = ["wsi_id", "patient_id", "label", "n_patches", "error"]
    pd.DataFrame(rows, columns=manifest_cols).to_csv(out_dir / "manifest.csv", index=False)
    pd.DataFrame(slide_rows, columns=slide_cols).to_csv(out_dir / "slides.csv", index=False)
    return pd.DataFrame(rows, columns=manifest_cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="DHMC MetaData.csv or detailed label CSV")
    ap.add_argument("--wsi-dir", required=True, help="directory containing the .tif WSIs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch-size", dest="patch_size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=None, help="default: patch_size")
    ap.add_argument("--level", type=int, default=0, help="WSI level (0 = full resolution)")
    ap.add_argument("--patches-per-slide", dest="patches_per_slide", type=int, default=32)
    ap.add_argument("--min-tissue-frac", dest="min_tissue_frac", type=float, default=0.5)
    ap.add_argument("--thumb-size", dest="thumb_size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-slides", dest="max_slides", type=int, default=None)
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="parse metadata only")
    args = ap.parse_args()

    meta = parse_metadata(pd.read_csv(args.metadata))
    if meta.empty:
        raise SystemExit("no labelled slides found in the metadata")
    print(meta["label"].value_counts().to_string())
    if args.dry_run:
        print(f"parsed {len(meta)} labelled slides")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = import_slides(
        meta,
        Path(args.wsi_dir),
        out,
        args.patch_size,
        args.stride or args.patch_size,
        args.level,
        args.patches_per_slide,
        args.min_tissue_frac,
        args.thumb_size,
        args.seed,
        args.max_slides,
    )
    print(f"wrote {len(df)} patches from {df['wsi_id'].nunique() if len(df) else 0} slides to {out}")
    if len(df):
        print(df.groupby("label")["roi_id"].count().to_string())
        missing = pd.read_csv(out / "slides.csv")
        missing = missing[missing["n_patches"] == 0]
        if len(missing):
            print(f"[warn] {len(missing)} slides produced no patches; see {out / 'slides.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
