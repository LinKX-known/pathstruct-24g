"""Turning ROI images into cell tables.

**The classical backend here is a debug tool.** On real H&E it under-segments touching
nuclei. For anything reported, run the official CellViT-Inference package
(`pip install cellvit`) and convert with `scripts/data/run_cellvit.py`; see
`docs/DEPENDENCIES.md` for the full list of what is borrowed and what is not.

The topology descriptor needs one thing: where the nuclei are. This module
produces that, with two routes:

``classical``
    Haematoxylin channel -> Otsu -> distance transform -> watershed. No weights,
    no GPU, runs anywhere. Quality is below a trained segmenter, but the
    *arrangement* signal survives: what matters for a lumen is whether nuclei
    form a ring, not whether the boundaries are pixel-perfect.

``mask``
    If you already ran CellViT++ / HoVer-Net / StarDist and saved an instance
    mask (one integer per nucleus), this converts it to centroids. That is the
    recommended route for real data: use the best segmenter you have, export
    masks, and let this repo own everything downstream.

Both write ``cells/{roi_id}.csv`` with ``x, y, type``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cells import write_cells


def segment_classical(
    image,
    min_area: int = 12,
    max_area: int = 5000,
    min_distance: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nuclei centroids from an RGB array via haematoxylin thresholding + watershed."""
    from scipy import ndimage
    from skimage.color import rgb2hed
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.measure import regionprops
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed

    rgb = np.asarray(image, dtype=np.float64)[..., :3] / 255.0
    haematoxylin = rgb2hed(rgb)[..., 0]

    lo, hi = np.quantile(haematoxylin, [0.01, 0.99])
    norm = np.clip((haematoxylin - lo) / max(hi - lo, 1e-6), 0, 1)

    try:
        thresh = threshold_otsu(norm)
    except ValueError:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)

    # Otsu returns a histogram bin centre.  When the background is a single
    # near-zero value and nuclei occupy a few percent of the pixels, the
    # selected centre can land just *below* that background value.  ``>`` then
    # calls the whole image foreground (observed on 4/36 synthetic ROIs: the
    # threshold was 0.0215 while the background normalised to 0.0231).  Detect
    # that degenerate case and split halfway between background and the upper
    # tail instead.
    if thresh <= float(np.median(norm)):
        thresh = 0.5 * (float(np.median(norm)) + float(np.quantile(norm, 0.99)))

    binary = norm > thresh
    try:  # scikit-image >= 0.26 renamed min_size -> max_size (<= semantics)
        binary = remove_small_objects(binary, max_size=min_area - 1)
    except TypeError:
        binary = remove_small_objects(binary, min_size=min_area)
    binary = ndimage.binary_fill_holes(binary)
    if not binary.any():
        return np.zeros((0, 2)), np.zeros(0, dtype=int)

    distance = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(
        distance, min_distance=min_distance, labels=binary, exclude_border=False
    )
    if len(coords) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)

    markers = np.zeros(binary.shape, dtype=int)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    labels = watershed(-distance, markers, mask=binary)

    points = []
    for prop in regionprops(labels):
        if min_area <= prop.area <= max_area:
            points.append((prop.centroid[1], prop.centroid[0]))
    if not points:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    pts = np.asarray(points, dtype=np.float64)
    return pts, np.zeros(len(pts), dtype=int)


def cells_from_instance_mask(mask: np.ndarray, min_area: int = 10, max_area: int = 5000):
    """Centroids from an integer instance mask (background = 0)."""
    from skimage.measure import regionprops

    labels = np.asarray(mask)
    if labels.ndim == 3:
        labels = labels[..., 0]
    points = []
    for prop in regionprops(labels.astype(int)):
        if min_area <= prop.area <= max_area:
            points.append((prop.centroid[1], prop.centroid[0]))
    if not points:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    pts = np.asarray(points, dtype=np.float64)
    return pts, np.zeros(len(pts), dtype=int)


def segment_image(image_path: str | Path, backend: str = "classical", **kwargs):
    from PIL import Image

    with Image.open(image_path) as im:
        if backend == "mask":
            # label maps must stay single-channel: do not convert to RGB
            arr = np.asarray(im)
        else:
            arr = np.asarray(im.convert("RGB"))
    if backend == "classical":
        return segment_classical(arr, **kwargs)
    if backend == "mask":
        return cells_from_instance_mask(arr, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; use 'classical' or 'mask'")


def segment_dir(
    images_dir: str | Path,
    out_dir: str | Path,
    backend: str = "classical",
    pattern: str = "*.png",
    **kwargs,
) -> pd.DataFrame:
    images = sorted(Path(images_dir).glob(pattern))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for img in images:
        roi_id = img.stem
        out_path = out_dir / f"{roi_id}.csv"
        # Remove a pre-existing table first.  Otherwise a failed segmentation
        # silently falls back to whatever was in the output directory (e.g. the
        # synthetic ground truth written by `pilot synth`), and doctor cannot
        # see that the segmenter failed.
        if out_path.exists():
            out_path.unlink()
        try:
            pts, types = segment_image(img, backend=backend, **kwargs)
        except Exception as exc:  # noqa: BLE001
            rows.append({"roi_id": roi_id, "n_cells": 0, "error": str(exc)[:120]})
            continue
        if len(pts) < 10:
            rows.append({"roi_id": roi_id, "n_cells": int(len(pts)), "error": "too few cells"})
            continue
        write_cells(pts, types, out_path)
        rows.append({"roi_id": roi_id, "n_cells": int(len(pts)), "error": ""})
    return pd.DataFrame(rows)

