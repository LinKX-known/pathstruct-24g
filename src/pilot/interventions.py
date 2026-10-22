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
    "collapse": "pulled together toward the centre of the region",
    "expand": "pushed apart away from the centre of the region",
    "translate": "shifted as a block",
}


def action_text(edit: "Edit", image_size: Optional[Tuple[float, float]] = None) -> str:
    """Text that fully specifies the edit, including the translation vector.

    The translation direction is part of the input, not the label: a block
    shifted across a bridge and the same block shifted along it can have
    different effects. Without the vector the task is not a function of the
    observation, and a text-only model is at chance.
    """
    if edit.kind != "translate":
        return ACTION_TEXT[edit.kind]
    vx, vy = edit.vector
    if image_size is not None:
        w, h = image_size
        vx = vx / max(float(w), 1e-9) * 1000.0
        vy = vy / max(float(h), 1e-9) * 1000.0
        return f"shifted as a block by ({vx:.1f}, {vy:.1f}) in 0-1000 image coordinates"
    return f"shifted as a block by ({vx:.2f}, {vy:.2f})"


def region_phrase(center: Tuple[float, float], radius: float, image_size=None) -> str:
    """A region spec in 0-1000 coordinates (the convention used in prompts)."""
    if image_size is not None:
        w, h = image_size
        cx = center[0] / max(float(w), 1e-9) * 1000.0
        cy = center[1] / max(float(h), 1e-9) * 1000.0
        r = radius / max(float(w), float(h), 1e-9) * 1000.0
        return (
            f"a circle centred at ({cx:.1f}, {cy:.1f}) with radius {r:.1f} "
            f"in 0-1000 image coordinates"
        )
    cx, cy = center
    return f"a circle centred at ({cx:.2f}, {cy:.2f}) with radius {radius:.2f}"


@dataclass
class Edit:
    edit_id: str
    kind: str
    center: Tuple[float, float]
    radius: float
    vector: Tuple[float, float] = (0.0, 0.0)
    affected: int = 0

    def prompt(self, image_size: Optional[Tuple[float, float]] = None) -> str:
        """A self-contained prompt for this edit (used by diagnostics/tests)."""
        return (
            f"Consider {region_phrase(self.center, self.radius, image_size)}. "
            f"Inside it, the cells are {action_text(self, image_size)}. "
            f"Would the number of connected cell groups or enclosed lumina change? "
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
    balance_by_kind: bool = True,
    min_affected: int = 0,
    max_attempts_factor: int = 40,
    return_points: bool = False,
) -> List[Intervention]:
    """Sample edits, balancing the yes/no label *within each edit kind*.

    Overall balance is not enough. If ``collapse`` changes topology 65% of the
    time while ``expand`` does so 41% of the time, a text-only model can read
    the edit kind from the prompt and reach ~0.59 balanced accuracy without
    looking at the image. Alternating yes/no within each kind removes that
    shortcut; a text-only model is then at 0.5.

    Balance matters: an unbalanced set can be solved by always answering the
    majority class, and the model would learn nothing about structure.

    ``min_affected`` rejects edits whose box contains fewer than that many
    cells. Empty boxes are always "no" (moving nothing changes nothing) and are
    a free visual shortcut; the feasibility run found 20% of training and 14%
    of validation boxes empty. Use ``min_affected=5`` for a structural
    benchmark and report the empty-box subset separately.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float64)
    out: List[Intervention] = []

    if not balance:
        for _ in range(max(1, n * max_attempts_factor)):
            if len(out) >= n:
                break
            edit = sample_edit(points, rng, radius_frac=float(rng.uniform(0.10, 0.22)))
            if edit.affected < min_affected:
                continue
            eff = measure_effect(points, types, edit, n_types=n_types, grid=grid)
            out.append(
                Intervention(
                    edit=edit,
                    effect=eff,
                    points_after=apply_edit(points, edit) if return_points else None,
                )
            )
        return out

    if balance_by_kind:
        slots: List[Tuple[str, bool]] = []
        per_kind: Dict[str, List[bool]] = {k: [] for k in EDIT_KINDS}
        for i in range(n):
            kind = EDIT_KINDS[i % len(EDIT_KINDS)]
            want = len(per_kind[kind]) % 2 == 0  # yes, no, yes, no ...
            per_kind[kind].append(want)
            slots.append((kind, want))
        per_slot_attempts = max(4, max_attempts_factor)
        for kind, want in slots:
            for _ in range(per_slot_attempts):
                edit = sample_edit(
                    points,
                    rng,
                    kind=kind,
                    radius_frac=float(rng.uniform(0.10, 0.22)),
                )
                if edit.affected < min_affected:
                    continue
                eff = measure_effect(points, types, edit, n_types=n_types, grid=grid)
                if eff.changed != want:
                    continue
                out.append(
                    Intervention(
                        edit=edit,
                        effect=eff,
                        points_after=apply_edit(points, edit) if return_points else None,
                    )
                )
                break
        return out

    yes = no = 0
    target_yes = n // 2
    target_no = n - target_yes
    for _ in range(max(1, n * max_attempts_factor)):
        if len(out) >= n or (yes >= target_yes and no >= target_no):
            break
        edit = sample_edit(points, rng, radius_frac=float(rng.uniform(0.10, 0.22)))
        if edit.affected < min_affected:
            continue
        eff = measure_effect(points, types, edit, n_types=n_types, grid=grid)
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
        "n_by_kind": {
            k: int(sum(1 for i in interventions if i.edit.kind == k)) for k in EDIT_KINDS
        },
    }

