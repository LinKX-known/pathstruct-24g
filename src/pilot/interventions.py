"""Structural interventions and their exact effects.

The training signal this module exists for is simple:

    show the model a region and an edit, ask whether the architecture changes.

The label is free and exact, because the edit is applied to the cell point set
and the descriptor is recomputed. Crucially, **the label is not determined by
the edit type**: collapsing a disk that sits in a lumen closes it, collapsing a
disk inside a solid sheet does nothing. A model cannot answer from the text of
the edit; it has to look at the local configuration.

Three edit families, all of which preserve the cell multiset (positions move,
counts do not change):

``collapse``   move cells inside a disk toward its centre  -> close a lumen
``expand``     move cells inside a disk outward           -> open a hole, break a bridge
``translate``  shift a disk of cells by a vector          -> split a component, cut a barrier
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .structures import Descriptor, _scales, describe, descriptor_distance, topology_changed

EDIT_KINDS = ("collapse", "expand", "translate")

ACTION_TEXT = {
    "collapse": "pulled together toward the centre",
    "expand": "pushed apart away from the centre",
    "translate": "shifted sideways",
}


@dataclass
class Edit:
    edit_id: str
    kind: str
    center: Tuple[float, float]
    radius: float
    vector: Tuple[float, float] = (0.0, 0.0)
    affected: int = 0

    def prompt(self) -> str:
        cx, cy = self.center
        return (
            f"A dashed circle is drawn at normalised position ({cx:.2f}, {cy:.2f}) "
            f"with radius {self.radius:.2f}. Inside it, the cells are "
            f"{ACTION_TEXT[self.kind]}. Does the tissue architecture in the image change? "
            f"Answer yes or no."
        )


@dataclass
class Effect:
    changed: bool
    delta: float
    before: Descriptor
    after: Descriptor


@dataclass
class Intervention:
    edit: Edit
    effect: Effect
    points_after: Optional[np.ndarray] = None


def _disk(points: np.ndarray, center: Tuple[float, float], radius: float) -> np.ndarray:
    d = np.linalg.norm(points - np.asarray(center)[None, :], axis=1)
    return d <= radius


def apply_edit(points: np.ndarray, edit: Edit) -> np.ndarray:
    p = np.array(points, dtype=np.float64, copy=True)
    m = _disk(p, edit.center, edit.radius)
    c = np.asarray(edit.center, dtype=np.float64)
    if edit.kind == "collapse":
        p[m] = c + (p[m] - c) * 0.12
    elif edit.kind == "expand":
        p[m] = c + (p[m] - c) * 2.4
    elif edit.kind == "translate":
        p[m] = p[m] + np.asarray(edit.vector, dtype=np.float64)
    else:
        raise ValueError(f"unknown edit kind {edit.kind}")
    return p


def sample_edit(
    points: np.ndarray,
    rng: np.random.Generator,
    kind: Optional[str] = None,
    radius_frac: float = 0.16,
    edit_id: str = "",
) -> Edit:
    points = np.asarray(points, dtype=np.float64)
    lo = np.quantile(points, 0.02, axis=0)
    hi = np.quantile(points, 0.98, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    kind = kind or str(rng.choice(EDIT_KINDS))
    center = lo + rng.random(2) * span
    radius = float(radius_frac * np.linalg.norm(span))
    vector = (0.0, 0.0)
    if kind == "translate":
        angle = rng.uniform(0, 2 * np.pi)
        mag = 0.45 * float(np.linalg.norm(span))
        vector = (float(mag * np.cos(angle)), float(mag * np.sin(angle)))
    affected = int(_disk(points, center, radius).sum())
    return Edit(
        edit_id=edit_id or f"{kind}_{rng.integers(0, 10**9)}",
        kind=kind,
        center=(float(center[0]), float(center[1])),
        radius=radius,
        vector=vector,
        affected=affected,
    )


def measure_effect(
    points: np.ndarray,
    types: Optional[np.ndarray],
    edit: Edit,
    n_types: int = 1,
    grid: int = 3,
) -> Effect:
    # cell size is a property of the specimen, not of the edited layout:
    # fixing both radii across the pair is what lets a collapse actually seal a lumen
    typ, gap = _scales(np.asarray(points, dtype=np.float64))
    kw = dict(n_types=n_types, grid=grid, cell_radius=1.3 * gap, comp_radius=1.5 * gap)
    before = describe(points, types, **kw)
    after = describe(apply_edit(points, edit), types, **kw)
    return Effect(
        changed=topology_changed(before, after),
        delta=descriptor_distance(before.values, after.values),
        before=before,
        after=after,
    )


def sample_interventions(
    points: np.ndarray,
    types: Optional[np.ndarray] = None,
    n: int = 8,
    seed: int = 0,
    n_types: int = 1,
    grid: int = 3,
    balance: bool = True,
    max_attempts_factor: int = 40,
    return_points: bool = False,
) -> List[Intervention]:
    """Sample edits, optionally balancing the yes/no label.

    Balance matters: an unbalanced set can be solved by always answering the
    majority class, and the model would learn nothing about structure.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float64)
    out: List[Intervention] = []
    yes = no = 0
    target_yes = n // 2 if balance else None
    target_no = n - (target_yes or 0) if balance else None

    for attempt in range(max(1, n * max_attempts_factor)):
        if len(out) >= n:
            break
        if balance and yes >= target_yes and no >= target_no:
            break
        edit = sample_edit(points, rng, radius_frac=float(rng.uniform(0.10, 0.22)))
        eff = measure_effect(points, types, edit, n_types=n_types, grid=grid)
        if balance:
            if eff.changed and yes >= target_yes:
                continue
            if (not eff.changed) and no >= target_no:
                continue
        out.append(
            Intervention(
                edit=edit,
                effect=eff,
                points_after=apply_edit(points, edit) if return_points else None,
            )
        )
        if eff.changed:
            yes += 1
        else:
            no += 1
    return out


def summarise(interventions: List[Intervention]) -> Dict[str, float]:
    if not interventions:
        return {"n": 0}
    changed = np.asarray([i.effect.changed for i in interventions], dtype=float)
    deltas = np.asarray([i.effect.delta for i in interventions], dtype=float)
    return {
        "n": len(interventions),
        "changed_rate": float(changed.mean()),
        "delta_median": float(np.nanmedian(deltas)),
        "by_kind": {
            k: float(np.mean([i.effect.changed for i in interventions if i.edit.kind == k]))
            for k in EDIT_KINDS
            if any(i.edit.kind == k for i in interventions)
        },
    }

