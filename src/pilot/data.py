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
from .interventions import sample_interventions
from .structures import describe

GROWTH_QUESTION = "What is the predominant growth pattern? Answer with one word."
LABELS = ("ring", "solid", "multifocal")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
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
def build_growth_examples(manifest: pd.DataFrame, question: str = GROWTH_QUESTION) -> List[Dict[str, Any]]:
    rows = []
    for r in manifest.to_dict(orient="records"):
        label = str(r.get("label", "")).strip()
        if not label or not r.get("image_path"):
            continue
        rows.append(
            {
                "task": "growth",
                "roi_id": r["roi_id"],
                "patient_id": r.get("patient_id", "") or r["roi_id"],
                "image": r["image_path"],
                "question": question,
                "answer": label,
            }
        )
    return rows


def _bbox_1000(center: Tuple[float, float], radius: float, size: Tuple[int, int]) -> List[float]:
    w, h = size
    x1 = max(0.0, (center[0] - radius) / max(w, 1) * 1000)
    y1 = max(0.0, (center[1] - radius) / max(h, 1) * 1000)
    x2 = min(1000.0, (center[0] + radius) / max(w, 1) * 1000)
    y2 = min(1000.0, (center[1] + radius) / max(h, 1) * 1000)
    return [round(v, 1) for v in (x1, y1, x2, y2)]


def build_cf_examples(
    manifest: pd.DataFrame,
    cells_dir: str | Path,
    per_roi: int = 6,
    seed: int = 0,
    n_types: int = 1,
) -> List[Dict[str, Any]]:
    from PIL import Image

    loaded = [load_cells_dir(cells_dir, rid) for rid in manifest["roi_id"]]
    types, vocab = global_type_index([t for _, t in [x for x in loaded if x is not None]])
    n_types = max(len(vocab), 1)

    rows: List[Dict[str, Any]] = []
    it = iter(types)
    for k, (row, item) in enumerate(zip(manifest.to_dict(orient="records"), loaded)):
        if item is None:
            continue
        points, _ = item
        t = next(it)
        if len(points) < 60:
            continue
        try:
            with Image.open(row["image_path"]) as im:
                size = im.size
        except Exception:
            continue

        desc = describe(points, t, n_types=n_types).values
        desc = [float(v) if np.isfinite(v) else 0.0 for v in desc]

        for i, itv in enumerate(
            sample_interventions(points, t, n=per_roi, seed=seed + k, n_types=n_types)
        ):
            e = itv.edit
            box = _bbox_1000(e.center, e.radius, size)
            rows.append(
                {
                    "task": "cf",
                    "ex_id": f"{row['roi_id']}__{i:03d}",
                    "roi_id": row["roi_id"],
                    "patient_id": row.get("patient_id", "") or row["roi_id"],
                    "image": row["image_path"],
                    "question": (
                        f"Consider the region [{', '.join(f'{v:.1f}' for v in box)}] "
                        f"(normalised 0-1000 coordinates). If the cells inside this region "
                        f"were moved, would the tissue architecture change? "
                        f"Answer with one word: yes or no."
                    ),
                    "answer": "yes" if itv.effect.changed else "no",
                    "changed": int(itv.effect.changed),
                    "delta": float(itv.effect.delta),
                    "kind": e.kind,
                    "center": [float(e.center[0]), float(e.center[1])],
                    "radius": float(e.radius),
                    "descriptor": desc,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
def make_synthetic(out_dir: str | Path, per_class: int = 12, n_cells: int = 220, size: int = 384, seed: int = 0):
    """Tiny cohort: three arrangement classes, with cell tables and images."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    img_dir = out / "images"
    cell_dir = out / "cells"
    img_dir.mkdir(parents=True, exist_ok=True)
    cell_dir.mkdir(parents=True, exist_ok=True)

    palette = np.array([[150, 90, 190], [230, 150, 170], [90, 140, 200], [80, 80, 80]], float)
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
            px = (pts + 1) / 2 * (size - 1)
            img = Image.new("RGB", (size, size), (247, 244, 240))
            draw = ImageDraw.Draw(img)
            for (x, y), tp in zip(px, types):
                c = tuple(int(v) for v in np.clip(palette[tp] + rng.normal(scale=6, size=3), 0, 255))
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c)
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
