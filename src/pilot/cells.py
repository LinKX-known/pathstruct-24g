"""Loading cell tables written by a nucleus segmenter (CellViT++, HoVer-Net, ...).

Expected file per ROI: ``{roi_id}.csv`` with columns ``x, y`` and optionally
``type`` / ``cell_type`` / ``label``. Coordinates are in pixels of the ROI image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TYPE_COLUMNS = ("type", "cell_type", "label", "class")


def load_cells(path: str | Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "x" not in cols or "y" not in cols:
        raise ValueError(f"{path}: need x and y columns, got {list(df.columns)}")
    points = df[[cols["x"], cols["y"]]].to_numpy(dtype=np.float64)
    types = None
    for name in TYPE_COLUMNS:
        if name in cols:
            types = df[cols[name]].astype(str).to_numpy()
            break
    return points, types


def load_cells_dir(
    cell_dir: str | Path,
    roi_id: str,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    path = Path(cell_dir) / f"{roi_id}.csv"
    if not path.exists():
        return None
    return load_cells(path)


def global_type_index(
    per_roi_types: Sequence[Optional[np.ndarray]],
) -> Tuple[List[Optional[np.ndarray]], List[str]]:
    """Map per-ROI string labels onto one shared integer vocabulary.

    Per-ROI factorisation would give the same string a different id in every
    ROI, which silently breaks any descriptor that groups by type.
    """
    vocab = sorted({t for arr in per_roi_types if arr is not None for t in arr})
    index = {t: i for i, t in enumerate(vocab)}
    out: List[Optional[np.ndarray]] = []
    for arr in per_roi_types:
        out.append(None if arr is None else np.asarray([index[t] for t in arr], dtype=int))
    return out, vocab

# --------------------------------------------------------------------------- #
# CellViT++ bridge
# --------------------------------------------------------------------------- #
def _centroid_of(cell: Dict) -> Optional[Tuple[float, float]]:
    for key in ("centroid", "center", "position"):
        value = cell.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
    geometry = cell.get("geometry") or {}
    coords = geometry.get("coordinates")
    if isinstance(coords, list) and coords:
        if isinstance(coords[0], (list, tuple)) and coords[0] and not isinstance(coords[0][0], (list, tuple)):
            return float(coords[0][0]), float(coords[0][1])
        if isinstance(coords[0], (list, tuple)) and coords[0] and isinstance(coords[0][0], (list, tuple)):
            return float(coords[0][0][0]), float(coords[0][0][1])
    return None


def _iter_cells(obj) -> Iterable[Dict]:
    if isinstance(obj, dict):
        for key in ("cells", "instances", "cell_dict"):
            if isinstance(obj.get(key), list):
                return obj[key]
        if any(k in obj for k in ("centroid", "geometry")):
            return [obj]
        return []
    if isinstance(obj, list):
        return obj
    return []


def cells_from_cellvit_json(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a CellViT++ ``*_cells.json`` export into centroids + class names.

    CellViT writes ``{"wsi_metadata": ..., "type_map": ..., "cells": [...]}``
    where each cell carries a ``centroid`` and a class label. This parser is
    deliberately tolerant about the surrounding schema: only the centroid and
    the class are needed downstream.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    rows = _iter_cells(data)
    points: List[Tuple[float, float]] = []
    labels: List[str] = []
    for cell in rows:
        if not isinstance(cell, dict):
            continue
        cen = _centroid_of(cell)
        if cen is None:
            continue
        label = cell.get("class") or cell.get("type") or cell.get("label") or "cell"
        points.append(cen)
        labels.append(str(label))
    if not points:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    vocab = {name: i for i, name in enumerate(sorted(set(labels)))}
    return np.asarray(points, dtype=np.float64), np.asarray([vocab[l] for l in labels], dtype=int)


def cells_from_cellvit_dir(json_dir: str | Path, out_dir: str | Path, suffix: str = "_cells") -> pd.DataFrame:
    """Convert every ``*_cells.json`` in a directory into ``cells/{roi_id}.csv``."""
    json_dir = Path(json_dir)
    rows = []
    for path in sorted(json_dir.glob(f"*{suffix}.json")):
        roi_id = path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem
        try:
            pts, types = cells_from_cellvit_json(path)
        except Exception as exc:  # noqa: BLE001
            rows.append({"roi_id": roi_id, "n_cells": 0, "error": str(exc)[:120]})
            continue
        if len(pts) == 0:
            rows.append({"roi_id": roi_id, "n_cells": 0, "error": "no centroids parsed"})
            continue
        write_cells(pts, types, Path(out_dir) / f"{roi_id}.csv")
        rows.append({"roi_id": roi_id, "n_cells": int(len(pts)), "error": ""})
    return pd.DataFrame(rows)

def write_cells(points: np.ndarray, types: np.ndarray, path: str | Path) -> None:
    """Write the one format the rest of the pipeline reads: x, y, type."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"x": points[:, 0], "y": points[:, 1], "type": [f"t{int(t)}" for t in types]}
    ).to_csv(path, index=False)
