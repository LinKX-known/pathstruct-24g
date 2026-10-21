"""Benchmark the segmentation stage on nucleus-annotated datasets.

PanNuke and MoNuSeg give something the pilot otherwise lacks: **ground-truth
nucleus instances**. That makes two questions answerable before any VLM is
trained:

1. How good is the segmenter we actually use (``classical`` by default)?
2. **Does the topology descriptor survive segmentation error?** If the lumen
   count flips when the segmenter is slightly wrong, the whole supervision
   signal is noise.

Both are cheap: PanNuke fold1 has ~2.7k 256x256 patches, so a full pass is
minutes on CPU.

Expected layout (standard releases)::

    images/  <stem>.png      RGB patch
    masks/   <stem>.png      instance mask, one integer per nucleus, 0 = bg

MoNuSeg's official annotations are XML coordinates; convert them to an instance
mask first (or use a mirror that ships PNG masks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .structures import describe, descriptor_distance


def load_instance_mask(path: str | Path, min_area: int = 10) -> np.ndarray:
    """Read an instance mask, tolerating RGB-encoded label maps."""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    labels = arr.astype(np.int64)
    if min_area > 0:
        # drop specks: relabel and keep components above the area threshold
        from scipy import ndimage

        keep = np.zeros(int(labels.max()) + 1, dtype=bool)
        counts = np.bincount(labels.ravel())
        keep[1:] = counts[1:] >= min_area
        labels = np.where(keep[labels], labels, 0)
    return labels


def centroids_from_mask(labels: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    present = np.unique(labels[labels > 0])
    if present.size == 0:
        return np.zeros((0, 2))
    # only labels that still exist: a removed speck would give a NaN centroid
    centers = ndimage.center_of_mass(labels > 0, labels, present)
    pts = np.asarray([(c[1], c[0]) for c in centers], dtype=np.float64)
    return pts[np.all(np.isfinite(pts), axis=1)]


def match_points(
    pred: np.ndarray,
    gt: np.ndarray,
    radius: float = 8.0,
) -> Tuple[int, int, int, np.ndarray]:
    """Greedy one-to-one matching within ``radius``. Returns tp, fp, fn, errors."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if len(pred):
        pred = pred[np.all(np.isfinite(pred), axis=1)]
    if len(gt):
        gt = gt[np.all(np.isfinite(gt), axis=1)]
    if len(gt) == 0:
        return 0, len(pred), 0, np.zeros(0)
    if len(pred) == 0:
        return 0, 0, len(gt), np.zeros(0)

    tree = cKDTree(pred)
    dist, idx = tree.query(gt, k=1)
    order = np.argsort(dist)
    used_pred = set()
    errors: List[float] = []
    for gi in order:
        if dist[gi] > radius:
            continue
        pi = int(idx[gi])
        if pi in used_pred:
            continue
        used_pred.add(pi)
        errors.append(float(dist[gi]))
    tp = len(errors)
    return tp, len(pred) - tp, len(gt) - tp, np.asarray(errors)


def benchmark(
    images_dir: str | Path,
    masks_dir: str | Path,
    segmenter=None,
    pattern: str = "*.png",
    match_radius: float = 8.0,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Run a segmenter over a patch set and compare against instance masks."""
    from .segment import segment_image

    segmenter = segmenter or (lambda p: segment_image(p, backend="classical"))
    images = sorted(Path(images_dir).glob(pattern))
    if limit:
        images = images[:limit]

    rows = []
    for img in images:
        mask_path = Path(masks_dir) / img.name
        if not mask_path.exists():
            continue
        try:
            labels = load_instance_mask(mask_path)
            gt = centroids_from_mask(labels)
            pred, _ = segmenter(img)
        except Exception as exc:  # noqa: BLE001
            rows.append({"stem": img.stem, "error": str(exc)[:120]})
            continue

        tp, fp, fn, errors = match_points(np.asarray(pred), gt, match_radius)
        d_gt = describe(gt) if len(gt) >= 8 else None
        d_pred = describe(np.asarray(pred)) if len(pred) >= 8 else None
        delta = (
            descriptor_distance(d_gt.values, d_pred.values)
            if (d_gt is not None and d_pred is not None)
            else float("nan")
        )
        rows.append(
            {
                "stem": img.stem,
                "n_gt": int(len(gt)),
                "n_pred": int(len(pred)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "centroid_err_median": float(np.median(errors)) if len(errors) else float("nan"),
                "descriptor_delta": delta,
                "hole_gt": float(d_gt.to_dict()["hole_count"]) if d_gt else float("nan"),
                "hole_pred": float(d_pred.to_dict()["hole_count"]) if d_pred else float("nan"),
                "error": "",
            }
        )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"n": 0}
    tp = float(df["tp"].sum())
    fp = float(df["fp"].sum())
    fn = float(df["fn"].sum())
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "n_patches": int(len(df)),
        "detection_f1": f1,
        "precision": precision,
        "recall": recall,
        "centroid_err_median": float(df["centroid_err_median"].median()),
        "descriptor_delta_median": float(df["descriptor_delta"].median()),
        "descriptor_delta_p90": float(df["descriptor_delta"].quantile(0.9)),
        "hole_agreement": float((df["hole_gt"] == df["hole_pred"]).mean()),
    }
