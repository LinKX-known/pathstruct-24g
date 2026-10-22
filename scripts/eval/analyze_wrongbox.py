#!/usr/bin/env python
"""Paired analysis of the wrong-box control.

The wrong-box set is built so that the same image and question are shown with
the green box moved to a region where the *recomputed* label is ``no``.  The
question is whether the model follows the box or keeps answering from global
texture / a box-position prior.

Key numbers
-----------
``wrongbox_yes_rate``
    Fraction of shifted-box rows answered ``yes``.  Since every kept row has
    gold ``no``, this is the shortcut rate (lower is better).
``flip_no_rate``
    Among rows whose original box was gold ``yes`` and whose shifted box is
    gold ``no``, the fraction answered ``no``.  This is the local-sensitivity
    rate: an honest model should flip.
``stay_yes_rate``
    The complement of ``flip_no_rate`` on the same subset.
``paired_logit_drop``
    Mean original logit minus shifted-box logit on the flip subset.  A model
    that uses the region should drop its yes-evidence when the region moves.

    python scripts/eval/analyze_wrongbox.py \
        --real out/heads_topo_real_s0.json \
        --wrongbox out/heads_wrongbox_real_s0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _rate(rows, key, value):
    if not rows:
        return float("nan")
    return float(np.mean([int(r[key]) == value for r in rows]))


def _metrics(pairs):
    if not pairs:
        return {"n": 0}
    flip = [(o, w) for o, w in pairs if int(o["changed"]) == 1 and int(w["changed"]) == 0]
    both_no = [(o, w) for o, w in pairs if int(o["changed"]) == 0 and int(w["changed"]) == 0]
    orig_yes = [(o, w) for o, w in pairs if int(o["changed"]) == 1]
    return {
        "n": len(pairs),
        "n_flip_yes_to_no": len(flip),
        "n_both_no": len(both_no),
        "n_orig_yes": len(orig_yes),
        "wrongbox_yes_rate": _rate([w for _, w in pairs], "pred", 1),
        "wrongbox_no_rate": _rate([w for _, w in pairs], "pred", 0),
        "orig_yes_rate": _rate([o for o, _ in orig_yes], "pred", 1),
        "flip_no_rate": _rate([w for _, w in flip], "pred", 0),
        "stay_yes_rate": _rate([w for _, w in flip], "pred", 1),
        "both_no_no_rate": _rate([w for _, w in both_no], "pred", 0),
        "both_no_yes_rate": _rate([w for _, w in both_no], "pred", 1),
        "paired_logit_drop": float(
            np.mean([float(o["logit"]) - float(w["logit"]) for o, w in flip])
        )
        if flip
        else float("nan"),
    }


def _group_bootstrap(pairs, n_boot=5000, seed=0):
    groups = np.asarray([o.get("patient_id", "") for o, _ in pairs])
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    keys = (
        "wrongbox_yes_rate",
        "flip_no_rate",
        "stay_yes_rate",
        "paired_logit_drop",
    )
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        sub = [pairs[i] for i in idx]
        m = _metrics(sub)
        for k in keys:
            samples[k].append(m.get(k, float("nan")))
    out = {}
    for k, vals in samples.items():
        a = np.asarray(vals, dtype=float)
        if np.isfinite(a).any():
            out[k] = [float(v) for v in np.nanpercentile(a, [2.5, 97.5])]
        else:
            out[k] = [float("nan"), float("nan")]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="original-box eval JSON")
    ap.add_argument("--wrongbox", required=True, help="shifted-box eval JSON")
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    original = {r["ex_id"]: r for r in read_rows(args.real)}
    shifted = read_rows(args.wrongbox)
    pairs = [(original[r["source_ex_id"]], r) for r in shifted if r.get("source_ex_id") in original]
    if not pairs:
        raise SystemExit("no paired rows; wrongbox rows need `source_ex_id` in the real file")

    result = {
        "n_real": len(original),
        "n_wrongbox": len(shifted),
        "n_pairs": len(pairs),
        "overall": _metrics(pairs),
        "overall_ci": _group_bootstrap(pairs, args.n_boot),
        "by_kind": {},
    }
    for kind in sorted({o.get("kind", "") for o, _ in pairs}):
        sub = [(o, w) for o, w in pairs if o.get("kind", "") == kind]
        result["by_kind"][kind] = _metrics(sub)

    print(json.dumps(result, indent=2, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
