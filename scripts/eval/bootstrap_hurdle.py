#!/usr/bin/env python
"""Patient-grouped bootstrap for a plain-vs-hurdle structure-head comparison.

Both prediction files must come from ``eval_heads.py --out`` on the same rows.
The plain run uses ``s_pred`` for presence and positive-value metrics; the
hurdle run uses its dedicated ``s_presence`` logits and ``s_reg`` conditional
regression.  Rows are paired by ``ex_id`` and resampled by ``patient_id`` so
that repeated patches from one slide do not count as independent evidence.

    python scripts/eval/bootstrap_hurdle.py \
        --old out/heads_topo_real_s0.json \
        --new out/heads_hurdle_real_s0.json \
        --dims 5 --n-boot 5000 --out out/bootstrap_hurdle_monuseg_s0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _auc(y, score):
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _positive_mae(rows, dim, key, positive):
    idx = np.flatnonzero(positive)
    if len(idx) == 0:
        return float("nan")
    return float(
        np.mean(
            [abs(float(rows[i][key][dim]) - np.log1p(float(rows[i]["descriptor"][dim]))) for i in idx]
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="plain/regression-only heads JSON")
    ap.add_argument("--new", required=True, help="hurdle heads JSON")
    ap.add_argument("--dims", type=int, default=5)
    ap.add_argument("--hurdle-dims", dest="hurdle_dims", default=None,
                    help="comma-separated; default: read from the new records")
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    old = {r["ex_id"]: r for r in read_rows(args.old)}
    new_rows = read_rows(args.new)
    pairs = [(old[r["ex_id"]], r) for r in new_rows if r["ex_id"] in old]
    if not pairs:
        raise SystemExit("no paired rows by ex_id")
    hurdle_dims = (
        [int(x) for x in args.hurdle_dims.split(",")]
        if args.hurdle_dims
        else [int(x) for x in new_rows[0].get("hurdle_dims", [])]
    )
    if not hurdle_dims:
        raise SystemExit("new run has no hurdle_dims; pass --hurdle-dims")

    groups = np.asarray([o.get("patient_id", "") for o, _ in pairs])
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(args.seed)

    def point(idx, dim):
        o = [pairs[i][0] for i in idx]
        n = [pairs[i][1] for i in idx]
        y = np.asarray([float(r["descriptor"][dim]) > 0 for r in o])
        j = hurdle_dims.index(dim)
        old_score = np.asarray([float(r["s_pred"][dim]) for r in o])
        new_score = np.asarray([float(r["s_presence"][j]) for r in n])
        return (
            _auc(y, old_score),
            _auc(y, new_score),
            _positive_mae(o, dim, "s_pred", y),
            _positive_mae(n, dim, "s_reg", y),
        )

    result = {
        "n_pairs": len(pairs),
        "n_groups": int(len(uniq)),
        "hurdle_dims": hurdle_dims,
        "dims": {},
    }
    for dim in hurdle_dims:
        boot = []
        for _ in range(args.n_boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([members[g] for g in pick])
            boot.append(point(idx, dim))
        a = np.asarray(boot, dtype=float)

        delta_auc = a[:, 1] - a[:, 0]
        delta_mae = a[:, 3] - a[:, 2]

        def ci(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            return [float(v) for v in np.percentile(x, [2.5, 97.5])] if len(x) else [float("nan")] * 2

        result["dims"][str(dim)] = {
            "old_presence_auc": float(np.nanmean(a[:, 0])),
            "new_presence_auc": float(np.nanmean(a[:, 1])),
            "delta_presence_auc": float(np.nanmean(delta_auc)),
            "old_positive_mae": float(np.nanmean(a[:, 2])),
            "new_positive_mae": float(np.nanmean(a[:, 3])),
            "delta_positive_mae": float(np.nanmean(delta_mae)),
            "ci": {
                "old_presence_auc": ci(a[:, 0]),
                "new_presence_auc": ci(a[:, 1]),
                "delta_presence_auc": ci(delta_auc),
                "old_positive_mae": ci(a[:, 2]),
                "new_positive_mae": ci(a[:, 3]),
                "delta_positive_mae": ci(delta_mae),
            },
        }
    print(json.dumps(result, indent=2, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
