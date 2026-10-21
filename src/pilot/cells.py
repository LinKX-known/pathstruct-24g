"""Loading cell tables written by a nucleus segmenter (CellViT++, HoVer-Net, ...).

Expected file per ROI: ``{roi_id}.csv`` with columns ``x, y`` and optionally
``type`` / ``cell_type`` / ``label``. Coordinates are in pixels of the ROI image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
