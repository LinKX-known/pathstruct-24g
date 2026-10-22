"""Building the three pilot datasets, plus a synthetic cohort for dry runs.

``growth``
    image + question -> one-word growth pattern. The downstream task.

``counterfactual``
    image + region + edit -> yes/no "does the architecture change". The
    auxiliary task, and the one that cannot be solved from text.

``synth``
    ring / solid / multifocal ROIs with cell tables, so the whole pipeline can
    be exercised on a laptop before any real data or GPU is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cells import global_type_index, load_cells_dir
from .interventions import action_text, sample_interventions
from .structures import describe

GROWTH_QUESTION = (
    "What is the predominant growth pattern? Choose one of: ring, solid, multifocal. "
    "Answer with one word."
)
LABELS = ("ring", "solid", "multifocal")

# Same palette as make_synthetic; kept here so the before/after renderer and the
# cohort generator cannot drift apart.
PALETTE = np.array([[150, 90, 190], [230, 150, 170], [90, 140, 200], [80, 80, 80]], float)
BACKGROUND = (247, 244, 240)


def render_cells(
    points: np.ndarray,
    types: Optional[np.ndarray],
    size: Tuple[int, int],
    dot_radius: float = 8.0,
    seed: int = 0,
    box: Optional[Tuple[float, float, float, float]] = None,
    label: str = "",
):
    """Render a cell point set as a clean diagram (synthetic, not H&E).

    Used for the before/after presentation: the model sees the same cell set
    before and after the edit, so it can compare topology instead of mentally
    simulating the move. The same seed is used for both panels, so each cell
    keeps its colour.
    """
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    w, h = int(size[0]), int(size[1])
    img = Image.new("RGB", (w, h), BACKGROUND)
    draw = ImageDraw.Draw(img)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if types is None:
        types = np.zeros(len(pts), dtype=int)
    types = np.asarray(types)
    if len(types) and not np.issubdtype(types.dtype, np.integer):
        # CellViT string labels: map to a stable local order for colour only.
        vocab = {name: i for i, name in enumerate(sorted(set(map(str, types))))}
        types = np.asarray([vocab[str(t)] for t in types], dtype=int)
    for (x, y), tp in zip(pts, types):
        c = tuple(
            int(v)
            for v in np.clip(PALETTE[int(tp) % len(PALETTE)] + rng.normal(scale=6, size=3), 0, 255)
        )
        draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill=c)
    if box is not None:
        lw = max(2, int(round(min(w, h) * 0.007)))
        draw.rectangle(
            [max(0, box[0]), max(0, box[1]), min(w - 1, box[2]), min(h - 1, box[3])],
            outline=(0, 190, 0),
            width=lw,
        )
    if label:
        draw.text((4, 4), label, fill=(0, 0, 0))
    return img


def side_by_side(left, right, sep: int = 4):
    """Place two same-height panels side by side with a thin separator."""
    from PIL import Image

    gap = Image.new("RGB", (sep, left.size[1]), (200, 200, 200))
    out = Image.new("RGB", (left.size[0] + sep + right.size[0], left.size[1]), BACKGROUND)
    out.paste(left, (0, 0))
    out.paste(gap, (left.size[0], 0))
    out.paste(right, (left.size[0] + sep, 0))
    return out


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read JSONL, or a single JSON array (the format `pilot eval --out` writes).

    `--compare` used to call this on a file written by `--out`, which is a JSON
    array; parsing it line by line raised at the first ``[``.
    """
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(stripped)
        return list(data) if isinstance(data, list) else []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "roi_id" not in df.columns:
        raise ValueError(f"{path}: manifest needs a roi_id column")
    return df


# --------------------------------------------------------------------------- #
def build_growth_examples(
    manifest: pd.DataFrame,
    question: str = GROWTH_QUESTION,
    cells_dir: str | Path | None = None,
    with_descriptor: bool = False,
    min_cells: int = 8,
) -> List[Dict[str, Any]]:
    """Build downstream examples, optionally with a topology descriptor.

    P2 needs the structure head to be supervised on the *same* patches as the
    downstream task, not only on counterfactual examples from another cohort.
    Passing ``cells_dir`` + ``with_descriptor=True`` attaches the same
    descriptor used by the counterfactual rows, computed in the image frame so
    it lines up with the structure head's 3x3 pooling.
    """
    records = manifest.to_dict(orient="records")
    descriptors: Dict[str, Dict[str, Any]] = {}
    if with_descriptor and cells_dir is not None:
        from PIL import Image

        loaded = [load_cells_dir(cells_dir, str(r["roi_id"])) for r in records]
        indexed_types, vocab = global_type_index(
            [item[1] if item is not None else None for item in loaded]
        )
        n_types = max(len(vocab), 1)
        for rec, item, types in zip(records, loaded, indexed_types):
            if item is None:
                continue
            points, _ = item
            if len(points) < min_cells:
                continue
            try:
                with Image.open(rec["image_path"]) as im:
                    width, height = im.size
            except Exception:
                continue
            desc = describe(
                points,
                types,
                n_types=n_types,
                frame=((0.0, 0.0), (float(width), float(height))),
            )
            values = [float(v) if np.isfinite(v) else 0.0 for v in desc.values]
            descriptors[str(rec["roi_id"])] = {
                "descriptor": values,
                "n_cells": int(len(points)),
                "descriptor_names": list(desc.names),
            }

    rows = []
    for r in records:
        label = str(r.get("label", "")).strip()
        if not label or not r.get("image_path"):
            continue
        ex = {
            "task": "growth",
            "ex_id": f"growth__{r['roi_id']}",
            "roi_id": r["roi_id"],
            "patient_id": r.get("patient_id", "") or r["roi_id"],
            "wsi_id": r.get("wsi_id", ""),
            "image": r["image_path"],
            "question": question,
            "answer": label,
        }
        extra = descriptors.get(str(r["roi_id"]))
        if extra:
            ex.update(extra)
        rows.append(ex)
    return rows


def _bbox_1000(center: Tuple[float, float], radius: float, size: Tuple[int, int]) -> List[float]:
    w, h = size
    x1 = max(0.0, (center[0] - radius) / max(w, 1) * 1000)
    y1 = max(0.0, (center[1] - radius) / max(h, 1) * 1000)
    x2 = min(1000.0, (center[0] + radius) / max(w, 1) * 1000)
    y2 = min(1000.0, (center[1] + radius) / max(h, 1) * 1000)
    return [round(v, 1) for v in (x1, y1, x2, y2)]


def annotate_edit(
    src: str | Path,
    dst: str | Path,
    center: Tuple[float, float],
    radius: float,
    vector: Tuple[float, float] = (0.0, 0.0),
    kind: str = "collapse",
) -> Path:
    """Draw the edit region (and translation arrow) on a copy of the ROI.

    The box is the visual grounding for the words "the region". Drawing it is
    not optional decoration: without it the model has to map 0-1000 text
    coordinates onto image patches, which is a separate (and much harder) task.
    The label is still not visible: the box only says *where* the edit happens,
    and the arrow only says *which way* the cells move. Whether that movement
    closes a lumen or breaks a bridge still requires looking at the cells.
    """
    from PIL import Image, ImageDraw

    src = Path(src)
    dst = Path(dst)
    with Image.open(src) as im:
        img = im.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    lw = max(2, int(round(min(w, h) * 0.007)))

    x1 = float(np.clip(center[0] - radius, 0, w - 1))
    y1 = float(np.clip(center[1] - radius, 0, h - 1))
    x2 = float(np.clip(center[0] + radius, 0, w - 1))
    y2 = float(np.clip(center[1] + radius, 0, h - 1))
    draw.rectangle([x1, y1, x2, y2], outline=(0, 190, 0), width=lw)

    if kind == "translate":
        cx, cy = float(np.clip(center[0], 0, w - 1)), float(np.clip(center[1], 0, h - 1))
        ex = float(np.clip(center[0] + vector[0], 0, w - 1))
        ey = float(np.clip(center[1] + vector[1], 0, h - 1))
        if abs(ex - cx) + abs(ey - cy) > 1:
            draw.line([cx, cy, ex, ey], fill=(255, 140, 0), width=lw)
            import math

            ang = math.atan2(ey - cy, ex - cx)
            L = max(8.0, min(w, h) * 0.035)
            for delta in (math.radians(155), -math.radians(155)):
                draw.line(
                    [ex, ey, ex + L * math.cos(ang + delta), ey + L * math.sin(ang + delta)],
                    fill=(255, 140, 0),
                    width=lw,
                )

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    return dst


def build_cf_examples(
    manifest: pd.DataFrame,
    cells_dir: str | Path,
    per_roi: int = 6,
    seed: int = 0,
    n_types: int = 1,
    annotate: bool = False,
    annotate_dir: str | Path | None = None,
    balance_by_kind: bool = True,
    min_affected: int = 0,
    min_cells: int = 60,
    presentation: str = "box",
    render_dot_radius: float = 8.0,
) -> List[Dict[str, Any]]:
    from PIL import Image

    if presentation not in ("box", "before_after"):
        raise ValueError(f"unknown presentation {presentation!r}")
    if presentation == "before_after" and annotate_dir is None:
        raise ValueError("presentation='before_after' requires annotate_dir")

    loaded = [load_cells_dir(cells_dir, rid) for rid in manifest["roi_id"]]
    valid = [x for x in loaded if x is not None]
    types, vocab = global_type_index([t for _, t in valid])
    n_types = max(len(vocab), 1)

    rows: List[Dict[str, Any]] = []
    it = iter(types)
    for k, (row, item) in enumerate(zip(manifest.to_dict(orient="records"), loaded)):
        if item is None:
            continue
        points, _ = item
        t = next(it)
        if len(points) < min_cells:
            continue
        try:
            with Image.open(row["image_path"]) as im:
                size = im.size
        except Exception:
            continue
        w, h = int(size[0]), int(size[1])
        if w <= 0 or h <= 0:
            continue

        # The structure target uses the image frame, so its 3x3 tiles line up
        # with the 3x3 pooling of image tokens in the structure head.
        desc = describe(
            points, t, n_types=n_types, frame=((0.0, 0.0), (float(w), float(h)))
        ).values
        desc = [float(v) if np.isfinite(v) else 0.0 for v in desc]

        for i, itv in enumerate(
            sample_interventions(
                points,
                t,
                n=per_roi,
                seed=seed + k,
                n_types=n_types,
                balance_by_kind=balance_by_kind,
                min_affected=min_affected,
                return_points=presentation == "before_after",
            )
        ):
            e = itv.edit
            ex_id = f"{row['roi_id']}__{i:03d}"
            box = _bbox_1000(e.center, e.radius, size)
            box_px = (
                max(0.0, float(e.center[0] - e.radius)),
                max(0.0, float(e.center[1] - e.radius)),
                min(float(w - 1), float(e.center[0] + e.radius)),
                min(float(h - 1), float(e.center[1] + e.radius)),
            )
            if presentation == "before_after":
                if itv.points_after is None:
                    continue
                Path(annotate_dir).mkdir(parents=True, exist_ok=True)
                before = render_cells(
                    points,
                    t,
                    (w, h),
                    dot_radius=render_dot_radius,
                    seed=seed + k * 1000 + i,
                    box=box_px,
                    label="before",
                )
                after = render_cells(
                    itv.points_after,
                    t,
                    (w, h),
                    dot_radius=render_dot_radius,
                    seed=seed + k * 1000 + i,
                    box=box_px,
                    label="after",
                )
                combined = side_by_side(before, after)
                image_path = str(Path(annotate_dir) / f"{ex_id}.png")
                combined.save(image_path)
                annotated = True
                image_size = [int(combined.size[0]), int(combined.size[1])]
                question = (
                    "The image shows the same tissue before (left) and after (right) an "
                    "edit inside the green box. Did the number of connected cell groups or "
                    "the number of enclosed lumina change? Answer with one word: yes or no."
                )
            else:
                image_path = row["image_path"]
                if annotate and annotate_dir is not None:
                    try:
                        image_path = str(
                            annotate_edit(
                                row["image_path"],
                                Path(annotate_dir) / f"{ex_id}.png",
                                e.center,
                                e.radius,
                                e.vector,
                                e.kind,
                            )
                        )
                    except Exception:
                        image_path = row["image_path"]

                # Deliberately no coordinates in the annotated prompt. Putting
                # the box in text lets a text-only model learn position priors
                # (measured: 0.69 balanced accuracy on held-out patients, above
                # the 0.65 P1 gate) and defeats the point of the pilot. The
                # green box and the orange translation arrow are the only place
                # the location lives. The `--no-annotate` debug mode falls back
                # to coordinates; it is not the main path and will have a text
                # shortcut.
                annotated = image_path != row["image_path"]
                if annotated:
                    region = "The green box in the image marks a region."
                    move = (
                        "shifted as a block in the direction of the orange arrow"
                        if e.kind == "translate"
                        else action_text(e, (w, h))
                    )
                else:
                    region = (
                        f"The region [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] "
                        f"in 0-1000 image coordinates marks a region."
                    )
                    move = action_text(e, (w, h))
                image_size = [w, h]
                question = (
                    f"{region} Inside that region, "
                    f"the cells are {move}. Treat this as a hypothetical edit; the "
                    f"rest of the tissue does not move. Would this change the tissue "
                    f"architecture? A change means the number of connected cell groups "
                    f"or the number of enclosed lumina changes. Answer with one word: "
                    f"yes or no."
                )

            # Location features the auxiliary head consumes are normalised to
            # the image, so they are comparable across ROIs of different size.
            cx, cy = float(e.center[0]) / w, float(e.center[1]) / h
            rn = float(e.radius) / max(w, h)
            vx, vy = float(e.vector[0]) / w, float(e.vector[1]) / h

            rows.append(
                {
                    "task": "cf",
                    "ex_id": ex_id,
                    "roi_id": row["roi_id"],
                    "patient_id": row.get("patient_id", "") or row["roi_id"],
                    "wsi_id": row.get("wsi_id", ""),
                    "image": image_path,
                    "source_image": row["image_path"],
                    "image_size": image_size,
                    "panel_size": [w, h],
                    "annotated": bool(annotated),
                    "presentation": presentation,
                    "question": question,
                    "answer": "yes" if itv.effect.changed else "no",
                    "changed": int(itv.effect.changed),
                    "delta": float(itv.effect.delta),
                    "kind": e.kind,
                    "center": [cx, cy],
                    "radius": rn,
                    "vector": [vx, vy],
                    "box_1000": box,
                    "descriptor": desc,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
def _group_of(row: Dict[str, Any], group_key: str) -> str:
    value = row.get(group_key) or row.get("patient_id") or row.get("roi_id") or ""
    return str(value)


def patient_splits(
    growth: Sequence[Dict[str, Any]],
    cf: Sequence[Dict[str, Any]],
    val_frac: float = 0.3,
    seed: int = 0,
    group_key: str = "patient_id",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Patient-disjoint train/val split for both tasks.

    Evaluating on the same ROIs used for training turns P1/P2 into memorisation
    checks. ROIs from one patient are correlated, so the split is by patient,
    not by ROI. The same held-out patients are used for growth and
    counterfactual rows, which keeps the two tasks comparable.
    """
    all_groups = sorted(
        {_group_of(r, group_key) for r in list(growth) + list(cf) if _group_of(r, group_key)}
    )
    if val_frac <= 0 or len(all_groups) < 2:
        return list(growth), [], list(cf), []

    rng = np.random.default_rng(seed)
    rng.shuffle(all_groups)
    n_val = min(len(all_groups) - 1, max(1, int(round(len(all_groups) * val_frac))))
    val_groups = set(all_groups[:n_val])

    def part(rows: Sequence[Dict[str, Any]]):
        train = [r for r in rows if _group_of(r, group_key) not in val_groups]
        val = [r for r in rows if _group_of(r, group_key) in val_groups]
        return train, val

    g_train, g_val = part(growth)
    c_train, c_val = part(cf)
    return g_train, g_val, c_train, c_val


def make_synthetic(
    out_dir: str | Path,
    per_class: int = 12,
    n_cells: int = 220,
    size: int = 384,
    seed: int = 0,
    dot_radius: float = 3.0,
):
    """Tiny cohort: three arrangement classes, with cell tables and images."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    img_dir = out / "images"
    cell_dir = out / "cells"
    img_dir.mkdir(parents=True, exist_ok=True)
    cell_dir.mkdir(parents=True, exist_ok=True)

    palette = PALETTE
    rows = []
    for label in LABELS:
        for rep in range(per_class):
            roi_id = f"{label}_{rep:03d}"
            if label == "ring":
                t = rng.uniform(0, 2 * np.pi, n_cells)
                r = rng.uniform(0.62, 0.78, n_cells)
                pts = np.c_[r * np.cos(t), r * np.sin(t)]
            elif label == "solid":
                t = rng.uniform(0, 2 * np.pi, n_cells)
                r = np.sqrt(rng.uniform(0, 0.72**2, n_cells))
                pts = np.c_[r * np.cos(t), r * np.sin(t)]
            else:
                centers = np.array([[-0.45, -0.4], [0.5, -0.25], [0.0, 0.55]])
                idx = rng.integers(0, 3, n_cells)
                pts = centers[idx] + rng.normal(scale=0.14, size=(n_cells, 2))

            types = rng.choice(4, size=n_cells, p=[0.5, 0.25, 0.15, 0.10])
            # Multifocal blobs can leave the [-1, 1] box by a few cells; clamp
            # so the cell table never claims a centroid outside the image.
            px = np.clip((pts + 1) / 2 * (size - 1), 0, size - 1)
            img = Image.new("RGB", (size, size), BACKGROUND)
            draw = ImageDraw.Draw(img)
            r_dot = float(dot_radius)
            for (x, y), tp in zip(px, types):
                c = tuple(int(v) for v in np.clip(palette[tp] + rng.normal(scale=6, size=3), 0, 255))
                draw.ellipse([x - r_dot, y - r_dot, x + r_dot, y + r_dot], fill=c)
            img_path = img_dir / f"{roi_id}.png"
            img.save(img_path)
            pd.DataFrame({"x": px[:, 0], "y": px[:, 1], "type": [f"t{k}" for k in types]}).to_csv(
                cell_dir / f"{roi_id}.csv", index=False
            )
            rows.append(
                {
                    "roi_id": roi_id,
                    "patient_id": f"pt_{label}_{rep // 2}",
                    "image_path": str(img_path),
                    "label": label,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    return df
