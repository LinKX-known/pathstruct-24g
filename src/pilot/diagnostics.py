"""Diagnostics: render what the pipeline sees, and preflight a cohort.

TopoSlide's preprocessing saves every intermediate map (point maps, distance
transforms, persistence diagrams, critical-point overlays) and writes a
``topo_done.txt`` per slide. That is the right instinct: when a run looks wrong,
the fastest way to find out why is to look at the picture, not at the loss.

``render_roi``
    one PNG with three panels: the cells, the rasterised sheet with detected
    lumina, and the connected components. If the lumen is in the wrong place,
    the segmenter is wrong -- not the model.

``doctor``
    preflight a manifest + cell directory before spending GPU hours: missing
    images, too few cells, non-finite descriptors, class balance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .cells import load_cells_dir
from .structures import _scales, describe, hole_mask

PANEL = 256


def _numpy_to_rgb(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return np.zeros((PANEL, PANEL, 3), dtype=np.uint8)
    lo, hi = np.quantile(a, [0.02, 0.98])
    a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    return (a * 255).astype(np.uint8)


def _fit(img: np.ndarray, size: int = PANEL) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def render_roi(
    image_path: str | Path,
    points: np.ndarray,
    out_path: str | Path,
    cell_radius: Optional[float] = None,
) -> Dict[str, Any]:
    """Three-panel diagnostic figure for one ROI."""
    from PIL import Image, ImageDraw

    pts = np.asarray(points, dtype=np.float64)
    if cell_radius is None:
        cell_radius = 1.3 * _scales(pts)[1] if len(pts) >= 8 else 1.0
    holes, mask = hole_mask(pts, cell_radius=cell_radius)

    panels = []

    # 1. original + cells
    try:
        with Image.open(image_path) as im:
            width, height = im.size
            base = im.convert("RGB").resize((PANEL, PANEL))
        draw = ImageDraw.Draw(base)
        if len(pts):
            # cells are in image pixels: scale into the panel, do not re-normalise
            xy = pts * np.array([PANEL / max(width, 1), PANEL / max(height, 1)])
            for x, y in xy:
                draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 40, 40))
        panels.append(np.asarray(base))
    except Exception:
        panels.append(np.zeros((PANEL, PANEL, 3), dtype=np.uint8))

    # 2. raster + lumina
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = (150, 150, 150)
    rgb[holes] = (220, 40, 40)
    panels.append(_fit(rgb))

    # 3. connected components
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask)
    rng = np.random.default_rng(0)
    colors = rng.integers(40, 255, size=(max(n, 1) + 1, 3)).astype(np.uint8)
    colors[0] = 0
    panels.append(_fit(colors[lab]))

    canvas = np.concatenate(panels[:3], axis=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)

    desc = describe(pts).to_dict() if len(pts) >= 8 else {}
    return {
        "path": str(out_path),
        "n_cells": int(len(pts)),
        "components": int(n),
        "hole_pixels": int(holes.sum()),
        **{k: float(v) for k, v in desc.items() if k in ("n_comp", "hole_count", "hole_area_frac")},
    }


def doctor(
    manifest,
    cells_dir: str | Path,
    max_report: int = 10,
) -> Dict[str, Any]:
    """Check a cohort before training and report actionable problems."""
    rows = manifest.to_dict(orient="records") if hasattr(manifest, "to_dict") else list(manifest)
    problems: List[str] = []
    counts = []
    missing_cells = missing_images = too_few = bad_desc = 0
    labels: Dict[str, int] = {}

    for r in rows:
        rid = str(r.get("roi_id", ""))
        img = str(r.get("image_path", ""))
        labels[str(r.get("label", ""))] = labels.get(str(r.get("label", "")), 0) + 1
        if not img or not Path(img).exists():
            missing_images += 1
        item = load_cells_dir(cells_dir, rid)
        if item is None:
            missing_cells += 1
            continue
        pts, _ = item
        counts.append(len(pts))
        if len(pts) < 60:
            too_few += 1
            continue
        d = describe(pts).values
        if not np.all(np.isfinite(d)):
            bad_desc += 1

    if missing_images:
        problems.append(f"{missing_images} ROIs have a missing image_path")
    if missing_cells:
        problems.append(f"{missing_cells} ROIs have no cells/{{roi_id}}.csv (run `pilot segment`)")
    if too_few:
        problems.append(f"{too_few} ROIs have fewer than 60 cells (topology will be noise)")
    if bad_desc:
        problems.append(f"{bad_desc} ROIs produced non-finite descriptors")
    if labels and max(labels.values()) > 0.8 * len(rows):
        problems.append(
            f"label imbalance: {max(labels.values())}/{len(rows)} are one class; balanced accuracy will be unstable"
        )

    return {
        "n_rois": len(rows),
        "labels": labels,
        "cells_median": float(np.median(counts)) if counts else 0.0,
        "cells_min": int(np.min(counts)) if counts else 0,
        "cells_max": int(np.max(counts)) if counts else 0,
        "problems": problems,
        "ok": not problems,
    }
