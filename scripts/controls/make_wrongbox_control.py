#!/usr/bin/env python
"""Build the sharper control: same image, wrong box.

``make_blank_control`` removes the cells entirely, so it tests "is there any
cell evidence?" but not "did the model look at the *right* region?".  This
script shifts the green box (and, for translate, the arrow) to a random
location that still contains cells, while keeping the image, question and edit
kind unchanged.  It then **recomputes the exact label** by applying the same
edit at the new location to the ROI's cell table.

The paired analysis is the point:

* original box gold=yes, shifted box gold=no → an honest local model should
  flip to ``no``; a global-texture/box-prior model keeps saying ``yes``.
* if the shifted box still changes topology, the label stays ``yes`` and the
  row is a normal (harder) positive, not a shortcut probe.

Only ``presentation="box"`` is supported.  The before/after composite contains
two panels and the after-cell coordinates are not stored in the JSONL, so a
shifted box would not correspond to a well-defined edit.

    python scripts/controls/make_wrongbox_control.py \
        --data data/real/monuseg_test/cf_bal.jsonl \
        --cells data/real/monuseg/cells \
        --out data/real/monuseg_test/wrongbox \
        --jsonl data/real/monuseg_test/cf_bal_wrongbox.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pilot.data import annotate_edit, read_jsonl, write_jsonl  # noqa: E402
from pilot.interventions import Edit, measure_effect  # noqa: E402


def _load_points(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    pts = df[["x", "y"]].to_numpy(dtype=np.float64)
    return pts[np.all(np.isfinite(pts), axis=1)]


def _box(cx: float, cy: float, radius: float, size):
    w, h = size
    return (
        max(0.0, cx - radius),
        max(0.0, cy - radius),
        min(float(w - 1), cx + radius),
        min(float(h - 1), cy + radius),
    )


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 1.0


def _bbox_1000(cx: float, cy: float, radius: float, size):
    w, h = size
    x1 = max(0.0, (cx - radius) / max(w, 1) * 1000.0)
    y1 = max(0.0, (cy - radius) / max(h, 1) * 1000.0)
    x2 = min(1000.0, (cx + radius) / max(w, 1) * 1000.0)
    y2 = min(1000.0, (cy + radius) / max(h, 1) * 1000.0)
    return [round(v, 1) for v in (x1, y1, x2, y2)]


def sample_candidate(
    points: np.ndarray,
    radius_px: float,
    original_center_px,
    size,
    rng: np.random.Generator,
    min_affected: int,
    min_center_shift: float,
    max_iou: float,
):
    """Draw one centre in the cell support, far enough from the original box."""
    w, h = size
    lo = np.quantile(points, 0.02, axis=0)
    hi = np.quantile(points, 0.98, axis=0)
    span = np.maximum(hi - lo, 1.0)
    orig_box = _box(original_center_px[0], original_center_px[1], radius_px, size)
    diag = max(float(np.hypot(w, h)), 1.0)
    cand = lo + rng.random(2) * span
    cand = np.clip(cand, [0.0, 0.0], [float(w - 1), float(h - 1)])
    shift = float(np.linalg.norm(cand - original_center_px)) / diag
    if shift < min_center_shift:
        return None, 0
    box = _box(cand[0], cand[1], radius_px, size)
    if _iou(orig_box, box) > max_iou:
        return None, 0
    affected = int(
        (np.linalg.norm(points - cand[None, :], axis=1) <= radius_px).sum()
    )
    if affected < min_affected:
        return None, 0
    return cand, affected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cells", required=True, help="directory of <roi_id>.csv cell tables")
    ap.add_argument("--out", required=True, help="directory for shifted-box images")
    ap.add_argument("--jsonl", required=True, help="output JSONL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-affected", dest="min_affected", type=int, default=5)
    ap.add_argument("--min-cells", dest="min_cells", type=int, default=8)
    ap.add_argument(
        "--min-center-shift",
        dest="min_center_shift",
        type=float,
        default=0.20,
        help="minimum shift as a fraction of the image diagonal",
    )
    ap.add_argument("--max-iou", dest="max_iou", type=float, default=0.10)
    ap.add_argument(
        "--select-label",
        dest="select_label",
        choices=("no", "flip", "any"),
        default="no",
        help=(
            "keep shifted edits whose recomputed label is no (default), differs "
            "from the original (flip), or any label (any)"
        ),
    )
    ap.add_argument("--attempts", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N rows (debug)")
    ap.add_argument(
        "--no-check-original",
        dest="check_original",
        action="store_false",
        help="skip recomputing the original label (sanity check)",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    if not rows:
        raise SystemExit("empty data")
    if args.limit:
        rows = rows[: args.limit]
    if any(r.get("presentation", "box") != "box" for r in rows):
        raise SystemExit(
            "wrong-box control currently supports presentation='box' only "
            "(before/after needs both panels' cell coordinates)"
        )

    cells_dir = Path(args.cells)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    result = []
    missing_cells = too_few = failed_sample = rejected_label = 0
    original_mismatch = 0
    flip_yes_no = flip_no_yes = both_yes = both_no = 0
    per_kind = {}

    for r in rows:
        roi_id = str(r.get("roi_id", ""))
        cell_path = cells_dir / f"{roi_id}.csv"
        if not cell_path.exists():
            missing_cells += 1
            continue
        points = _load_points(cell_path)
        if len(points) < args.min_cells:
            too_few += 1
            continue

        w, h = int(r["image_size"][0]), int(r["image_size"][1])
        radius_px = float(r["radius"]) * max(w, h)
        original_center_px = np.asarray(
            [r["center"][0] * w, r["center"][1] * h], dtype=np.float64
        )
        vector_px = (
            float(r.get("vector", [0.0, 0.0])[0]) * w,
            float(r.get("vector", [0.0, 0.0])[1]) * h,
        )
        kind = str(r.get("kind", "collapse"))

        if args.check_original:
            orig_edit = Edit(
                edit_id=f"{r['ex_id']}_orig",
                kind=kind,
                center=tuple(original_center_px),
                radius=radius_px,
                vector=vector_px,
            )
            orig_effect = measure_effect(points, None, orig_edit)
            if bool(orig_effect.changed) != bool(r["changed"]):
                original_mismatch += 1

        cand = affected = edit = effect = None
        changed = None
        for _ in range(max(args.attempts, 1)):
            c, a = sample_candidate(
                points,
                radius_px,
                original_center_px,
                (w, h),
                rng,
                args.min_affected,
                args.min_center_shift,
                args.max_iou,
            )
            if c is None:
                continue
            e = Edit(
                edit_id=f"{r['ex_id']}_wrongbox",
                kind=kind,
                center=(float(c[0]), float(c[1])),
                radius=radius_px,
                vector=vector_px,
                affected=a,
            )
            eff = measure_effect(points, None, e)
            ch = bool(eff.changed)
            if args.select_label == "no" and ch:
                rejected_label += 1
                continue
            if args.select_label == "flip" and ch == bool(r["changed"]):
                rejected_label += 1
                continue
            cand, affected, edit, effect, changed = c, a, e, eff, ch
            break
        if cand is None:
            failed_sample += 1
            continue

        src = Path(r.get("source_image") or r["image"])
        dst = out_dir / f"{r['ex_id']}__wrongbox.png"
        annotate_edit(src, dst, edit.center, edit.radius, edit.vector, kind)

        out = dict(r)
        out.update(
            {
                "ex_id": f"{r['ex_id']}__wrongbox",
                "source_ex_id": r["ex_id"],
                "image": str(dst),
                "source_image": str(src),
                "center": [float(cand[0]) / w, float(cand[1]) / h],
                "radius": float(radius_px / max(w, h)),
                "vector": [vector_px[0] / w, vector_px[1] / h],
                "changed": int(changed),
                "answer": "yes" if changed else "no",
                "delta": float(effect.delta),
                "box_1000": _bbox_1000(cand[0], cand[1], radius_px, (w, h)),
                "wrongbox": True,
                "wrongbox_affected": affected,
                "original_changed": int(r["changed"]),
                "original_center": r["center"],
            }
        )
        # The descriptor is a property of the *before* cell table, not of the
        # box, so keep the original one.
        result.append(out)

        orig_yes = bool(r["changed"])
        if orig_yes and not changed:
            flip_yes_no += 1
        elif not orig_yes and changed:
            flip_no_yes += 1
        elif orig_yes and changed:
            both_yes += 1
        else:
            both_no += 1
        stats = per_kind.setdefault(kind, {"n": 0, "yes": 0, "flip_yes_no": 0})
        stats["n"] += 1
        stats["yes"] += int(changed)
        stats["flip_yes_no"] += int(orig_yes and not changed)

    write_jsonl(result, args.jsonl)
    n = len(result)
    print(
        f"wrote {n} wrong-box examples to {args.jsonl} "
        f"(skipped: missing cells={missing_cells}, too few cells={too_few}, "
        f"no valid shift/label={failed_sample}; rejected by --select-label={rejected_label})"
    )
    if n:
        print(
            f"gold after shift: yes={sum(r['changed'] for r in result) / n:.3f} | "
            f"original yes→shifted no={flip_yes_no} | "
            f"original no→shifted yes={flip_no_yes} | "
            f"both yes={both_yes} | both no={both_no}"
        )
        for kind, s in sorted(per_kind.items()):
            print(
                f"  {kind:9s}: n={s['n']:3d} shifted yes={s['yes'] / max(s['n'], 1):.3f} "
                f"yes→no flips={s['flip_yes_no']}"
            )
    if args.check_original and original_mismatch:
        print(
            f"[warn] recomputing the original label disagreed on "
            f"{original_mismatch}/{len(rows)} rows; the cell table and the box "
            "coordinates may use different units",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
