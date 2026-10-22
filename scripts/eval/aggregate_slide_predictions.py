#!/usr/bin/env python
"""Aggregate patch predictions to a slide answer and score them.

DHMC ships a slide-level predominant growth pattern, not patch labels.  The
practical P2 protocol is therefore patch sampling + patch classifier +
majority vote (or MIL).  This script does the majority-vote half and reports
slide-level metrics with a slide-grouped bootstrap, which is the unit the
labels actually live at.

Input is the JSON array written by ``pilot eval --out`` (or a JSONL with the
same rows).  Each row needs ``gold``, ``pred`` and one of ``wsi_id`` /
``patient_id`` / ``roi_id``.

    python scripts/eval/aggregate_slide_predictions.py \
        --pred out/dhmc_aux_eval.json --out out/dhmc_aux_slide.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def slide_key(row):
    return (
        str(row.get("wsi_id") or "")
        or str(row.get("patient_id") or "")
        or str(row.get("roi_id") or "").rsplit("_p", 1)[0]
    )


def aggregate(rows, min_patches=1, tie_break="first"):
    groups = defaultdict(list)
    for row in rows:
        groups[slide_key(row)].append(row)
    slides = []
    for key, items in sorted(groups.items()):
        if len(items) < min_patches:
            continue
        golds = Counter(str(r.get("gold", "")) for r in items if r.get("gold"))
        gold = golds.most_common(1)[0][0] if golds else ""
        preds = Counter(str(r["pred"]) for r in items if r.get("pred"))
        parse_failures = sum(1 for r in items if not r.get("pred"))
        if preds:
            top = preds.most_common()
            best = top[0][1]
            tied = [p for p, c in top if c == best]
            if len(tied) > 1:
                if tie_break == "error":
                    pred = ""
                else:
                    pred = sorted(tied)[0]
                tie = True
            else:
                pred = tied[0]
                tie = False
        else:
            pred = ""
            tie = False
        slides.append(
            {
                "slide": key,
                "gold": gold,
                "pred": pred,
                "n_patches": len(items),
                "parse_failures": parse_failures,
                "tie": tie,
                "vote_counts": dict(preds),
            }
        )
    return slides


def score(slides, labels=None):
    if not slides:
        return {"n_slides": 0}
    if labels is None:
        labels = sorted({s["gold"] for s in slides} | {s["pred"] for s in slides})
    labels = [lab for lab in labels if lab]
    tp = Counter()
    fp = Counter()
    fn = Counter()
    support = Counter()
    correct = 0
    for s in slides:
        support[s["gold"]] += 1
        if s["pred"] == s["gold"]:
            correct += 1
            tp[s["gold"]] += 1
        else:
            if s["pred"]:
                fp[s["pred"]] += 1
            fn[s["gold"]] += 1
    per_class = {}
    for lab in labels:
        prec = tp[lab] / max(tp[lab] + fp[lab], 1e-9)
        rec = tp[lab] / max(tp[lab] + fn[lab], 1e-9)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[lab] = {
            "n": int(support[lab]),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
    recalls = [v["recall"] for v in per_class.values() if v["n"] > 0]
    f1s = [v["f1"] for v in per_class.values() if v["n"] > 0]
    return {
        "n_slides": len(slides),
        "accuracy": float(correct / len(slides)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "parse_failures": int(sum(s["parse_failures"] for s in slides)),
        "ties": int(sum(s["tie"] for s in slides)),
        "per_class": per_class,
    }


def _paired_bootstrap(a_slides, b_slides, n_boot=5000, seed=0):
    a = {s["slide"]: s["pred"] == s["gold"] for s in a_slides}
    b = {s["slide"]: s["pred"] == s["gold"] for s in b_slides}
    keys = sorted(set(a) & set(b))
    if not keys:
        return {}
    x = np.asarray([a[k] for k in keys], dtype=float)
    y = np.asarray([b[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        diffs[i] = x[idx].mean() - y[idx].mean()
    return {
        "n": len(keys),
        "delta": float(x.mean() - y.mean()),
        "ci": [float(v) for v in np.percentile(diffs, [2.5, 97.5])],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--compare", default=None, help="second prediction file (e.g. no-aux)")
    ap.add_argument("--min-patches", dest="min_patches", type=int, default=1)
    ap.add_argument("--tie-break", dest="tie_break", choices=("first", "error"), default="first")
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    slides = aggregate(read_rows(args.pred), args.min_patches, args.tie_break)
    result = {"pred": args.pred, "slides": slides, "score": score(slides)}
    print(json.dumps({k: v for k, v in result.items() if k != "slides"}, indent=2, default=float))
    if args.compare:
        other = aggregate(read_rows(args.compare), args.min_patches, args.tie_break)
        result["compare"] = {
            "pred": args.compare,
            "score": score(other),
            "paired": _paired_bootstrap(slides, other, args.n_boot),
        }
        print(
            json.dumps(
                {k: v for k, v in result["compare"].items()},
                indent=2,
                default=float,
            )
        )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
