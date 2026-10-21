"""Structure descriptors: what is present, where, and how it is arranged.

Two blocks, concatenated:

``topology``
    Connectivity and lumina. ``n_comp`` is the number of connected components of
    the cell graph at 1.5 nearest-neighbour spacings (multifocality).
    ``hole_count`` / ``hole_area_frac`` come from rasterising the cells and
    counting holes in the resulting sheet -- the operational definition of a
    gland lumen, and far more stable than the cycle rank of a kNN graph, which
    is dominated by local triangulation.

``regional``
    A coarse grid of cell-type composition. This block is what forces a model to
    localise: image-wide counts can be matched without knowing where anything
    is, a per-tile composition cannot.

The descriptors are used for three things: regression targets for the structure
head, the definition of "did this intervention change the architecture", and the
quantity the probe measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .topology_h0 import h0_deaths

GRID = 3
RASTER_RES = 96
COMPONENT_RADIUS = 1.5


class _UnionFind:
    __slots__ = ("parent", "count")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.count = n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            self.count -= 1


def component_count(points: np.ndarray, radius: float) -> int:
    x = np.asarray(points, dtype=np.float64)
    pairs = cKDTree(x).query_pairs(radius, output_type="ndarray")
    uf = _UnionFind(len(x))
    for a, b in pairs:
        uf.union(int(a), int(b))
    return uf.count


def loop_count(points: np.ndarray, radius: float) -> int:
    """Cycle rank of the radius graph. Kept for diagnostics; not a descriptor."""
    x = np.asarray(points, dtype=np.float64)
    pairs = cKDTree(x).query_pairs(radius, output_type="ndarray")
    uf = _UnionFind(len(x))
    for a, b in pairs:
        uf.union(int(a), int(b))
    return int(len(pairs) - len(x) + uf.count)


def _scales(points: np.ndarray) -> Tuple[float, float]:
    """(typical, extreme) cell spacing.

    ``typical`` is the mean nearest-neighbour distance and sets the stroke width
    used when rasterising a cell. ``extreme`` is the 99th percentile and sets the
    connectivity radius, so that a single large gap does not shatter an
    otherwise connected ring into segments.
    """
    d, _ = cKDTree(points).query(points, k=2)
    d1 = d[:, 1]
    return float(d1.mean()), float(np.quantile(d1, 0.99))


def raster_mask(
    points: np.ndarray,
    resolution: int = RASTER_RES,
    cell_radius: Optional[float] = None,
) -> np.ndarray:
    """Binary sheet of cells, each drawn as a disk.

    The frame is expanded by the cell radius on every side *before* mapping, so
    dilation never gets clipped at the border. Without this a ring that fills
    the frame is cut into arcs, the lumen leaks into the background, and the
    hole count silently drops to zero -- which is exactly what happened on the
    synthetic ring until TopoSlide's padding trick was copied over.
    """
    x = np.asarray(points, dtype=np.float64)
    if len(x) == 0:
        return np.zeros((resolution, resolution), dtype=bool)
    if cell_radius is None:
        cell_radius = 1.3 * _scales(x)[1]
    lo = x.min(axis=0) - cell_radius
    hi = x.max(axis=0) + cell_radius
    span = np.maximum(hi - lo, 1e-9)
    scale = (resolution - 3) / float(span.max())
    r = max(1, int(round(cell_radius * scale)))
    xy = ((x - lo) * scale + 1.5).astype(int)
    xy = np.clip(xy, r + 1, resolution - r - 2)

    mask = np.zeros((resolution, resolution), dtype=bool)
    mask[xy[:, 1], xy[:, 0]] = True
    struct = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    struct[yy**2 + xx**2 <= r**2 + 0.5] = True
    return ndimage.binary_dilation(mask, structure=struct)


def hole_mask(
    points: np.ndarray,
    resolution: int = RASTER_RES,
    cell_radius: Optional[float] = None,
    close_px: int = 1,
    pad: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean lumen mask, plus the cell sheet it came from.

    Two details borrowed from TopoSlide's preprocessing:

    * **pad** before filling, so a structure touching the image border is
      connected to the outside instead of being counted as a lumen;
    * **close** one pixel before filling, because a single-pixel channel between
      two cells otherwise lets an obvious gland leak into the background. This
      was reproduced on a synthetic ring at 0.8-1.0x the gap scale, where plain
      filling reported zero lumina.
    """
    mask = raster_mask(points, resolution=resolution, cell_radius=cell_radius)
    if not mask.any():
        return np.zeros_like(mask), mask
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    if close_px > 0:
        k = 2 * close_px + 1
        padded = ndimage.binary_closing(padded, structure=np.ones((k, k), dtype=bool))
    filled = ndimage.binary_fill_holes(padded)
    holes = filled & ~padded
    return holes[pad:-pad, pad:-pad], mask


def hole_stats(
    points: np.ndarray,
    resolution: int = RASTER_RES,
    min_area_frac: float = 0.0015,
    cell_radius: Optional[float] = None,
) -> Tuple[int, float]:
    """Number and area fraction of lumina in the cell sheet."""
    holes, mask = hole_mask(points, resolution=resolution, cell_radius=cell_radius)
    if not holes.any():
        return 0, 0.0
    labels, n = ndimage.label(holes)
    if n == 0:
        return 0, 0.0
    areas = np.asarray(ndimage.sum(holes, labels, range(1, n + 1)), dtype=np.float64)
    min_area = max(4.0, min_area_frac * mask.size)
    return int((areas >= min_area).sum()), float(holes.sum() / max(mask.size, 1))


def topology_summary(points: np.ndarray, cell_radius: Optional[float] = None, comp_radius: Optional[float] = None) -> Dict[str, float]:
    """Scale-free topology of one cell population. Empty when there are too few cells."""
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or len(x) < 8:
        return {}
    typ, gap = _scales(x)
    if not np.isfinite(typ) or typ <= 0:
        return {}
    span = float(np.max(x.max(axis=0) - x.min(axis=0)))
    # a connectivity radius should never jump a large fraction of the field
    r_comp = float(comp_radius) if comp_radius is not None else min(1.5 * gap, 0.3 * span)
    r_cell = float(cell_radius) if cell_radius is not None else 1.3 * gap
    n_comp = float(component_count(x, r_comp))
    n_holes, hole_frac = hole_stats(x, cell_radius=r_cell)
    deaths = h0_deaths(x / typ)
    return {
        "n_comp": n_comp,
        "hole_count": float(n_holes),
        "hole_area_frac": hole_frac,
        "h0_max": float(deaths.max()) if len(deaths) else 0.0,
        "h0_sum": float(deaths.sum()) if len(deaths) else 0.0,
    }


TOPO_KEYS = ["n_comp", "hole_count", "hole_area_frac", "h0_max", "h0_sum"]


def _robust_unit(points: np.ndarray) -> np.ndarray:
    lo = np.quantile(points, 0.02, axis=0)
    hi = np.quantile(points, 0.98, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return np.clip((points - lo) / span, -0.2, 1.2)


def regional_descriptor(
    points: np.ndarray,
    types: Optional[np.ndarray] = None,
    n_types: int = 1,
    grid: int = GRID,
) -> np.ndarray:
    """Per-tile cell-type composition, as fractions of all cells."""
    x = np.asarray(points, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(grid * grid * (1 + n_types))
    u = _robust_unit(x)
    cell = np.clip((u * grid).astype(int), 0, grid - 1)
    flat = cell[:, 0] * grid + cell[:, 1]
    if types is None:
        types = np.zeros(n, dtype=int)
    types = np.asarray(types).astype(int)

    out = np.zeros(grid * grid * (1 + n_types))
    for t in range(n_types):
        out[t * grid * grid : (t + 1) * grid * grid] = (
            np.bincount(flat[types == t], minlength=grid * grid) / n
        )
    out[n_types * grid * grid :] = np.bincount(flat, minlength=grid * grid) / n
    return out


def descriptor_names(n_types: int = 1, grid: int = GRID) -> List[str]:
    names = list(TOPO_KEYS)
    for t in range(n_types):
        names += [f"reg_t{t}_g{i}" for i in range(grid * grid)]
    names += [f"reg_all_g{i}" for i in range(grid * grid)]
    return names


def descriptor_dim(n_types: int = 1, grid: int = GRID) -> int:
    return len(TOPO_KEYS) + grid * grid * (1 + n_types)


@dataclass
class Descriptor:
    values: np.ndarray
    names: List[str]

    def to_dict(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.values)}


def describe(
    points: np.ndarray,
    types: Optional[np.ndarray] = None,
    n_types: int = 1,
    grid: int = GRID,
    cell_radius: Optional[float] = None,
    comp_radius: Optional[float] = None,
) -> Descriptor:
    topo = topology_summary(points, cell_radius=cell_radius, comp_radius=comp_radius)
    if not topo:
        vec = np.full(descriptor_dim(n_types, grid), np.nan)
    else:
        vec = np.concatenate(
            [
                np.asarray([topo[k] for k in TOPO_KEYS], dtype=np.float64),
                regional_descriptor(points, types, n_types=n_types, grid=grid),
            ]
        )
    return Descriptor(values=vec, names=descriptor_names(n_types, grid))


def topology_changed(before: Descriptor, after: Descriptor, tol: float = 0.5) -> bool:
    """Did the intervention change connectivity or the number of lumina?

    ``tol`` is deliberately coarse: one lumen appearing or disappearing is a
    material change, a hole boundary moving by a pixel is not.
    """
    b = before.to_dict()
    a = after.to_dict()
    if any(k not in a or k not in b for k in ("n_comp", "hole_count")):
        return False
    if abs(a["n_comp"] - b["n_comp"]) >= 1:
        return True
    if abs(a["hole_count"] - b["hole_count"]) >= 1:
        return True
    if abs(a["hole_area_frac"] - b["hole_area_frac"]) >= tol * max(b["hole_area_frac"], 0.02):
        return True
    return False


def descriptor_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference, so distances are comparable across descriptor lengths."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def gudhi_dtm_summary(points: np.ndarray, k: int = 16) -> Dict[str, float]:
    """Optional exact DTM-Rips H0/H1 via gudhi (the filtration TopoSlide uses)."""
    try:
        import gudhi
    except ImportError:
        return {}
    from gudhi.dtm_rips_complex import DTMRipsComplex

    x = np.asarray(points, dtype=np.float64)
    if len(x) < k + 2:
        return {}
    st = DTMRipsComplex(points=x, k=k).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    dgm = st.persistence()
    h0 = np.asarray([b - a for dim, (a, b) in dgm if dim == 0 and np.isfinite(b)])
    h1 = np.asarray([b - a for dim, (a, b) in dgm if dim == 1 and np.isfinite(b)])
    return {
        "dtm_h0_sum": float(h0.sum()) if len(h0) else 0.0,
        "dtm_h0_max": float(h0.max()) if len(h0) else 0.0,
        "dtm_h1_sum": float(h1.sum()) if len(h1) else 0.0,
        "dtm_h1_count": float(len(h1)),
    }




