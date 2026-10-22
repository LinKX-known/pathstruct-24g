#!/usr/bin/env python
"""Patient/tissue-grouped bootstrap for the trained-head evaluation.

Consumes the JSON arrays written by ``scripts/eval/eval_heads.py --out`` (real
and, optionally, blank-box control) and reports:

* balanced accuracy / AUC with a 95% group bootstrap;
* real-minus-blank paired difference (same ex_id);
* a robust structure summary with group CIs.  The old headline ``mean R^2``
  over five dimensions is still printed as ``r2_mean_all`` for continuity, but
  it is dominated by zero-inflated dimensions such as ``hole_area_frac``.
  Prefer ``r2_median_all``, ``r2_mean_nondegenerate``, ``skill_mean`` and
  ``spearman_mean``; see ``pilot.structure_eval``.

    python scripts/eval/bootstrap_heads.py \
        --real out/heads_real_s0.json --blank out/heads_blank_s0.json \
        --train-data data/synth/cf_train_bal.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def balanced_acc(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    vals = []
    for c in (0, 1):
        m = y == c
        if m.any():
            vals.append(float(np.mean(pred[m] == y[m])))
    return float(np.mean(vals)) if vals else float("nan")


def auc(y, score):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


def _arrays(rows):
    true = np.asarray([r["descriptor"] for r in rows], dtype=float)
    pred = np.asarray([r["s_pred"] for r in rows], dtype=float)
    d = min(pred.shape[1], true.shape[1])
    return true[:, :d], pred[:, :d], d


def structure_summary(rows, train_mean=None):
    from pilot.structure_eval import log1p_target, per_dim_diagnostics, summary

    true, pred, d = _arrays(rows)
    diags = per_dim_diagnostics(
        log1p_target(true, d), pred, train_mean=train_mean
    )
    return summary(diags)


def _nan_ci(values, name):
    a = np.asarray([v.get(name, np.nan) for v in values], dtype=float)
    if not np.isfinite(a).any():
        return [float("nan"), float("nan")]
    return [float(v) for v in np.nanpercentile(a, [2.5, 97.5])]


def group_bootstrap(rows, train_mean=None, n_boot=5000, seed=0):
    groups = np.asarray([r["patient_id"] for r in rows])
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    bal, aucs = [], []
    struct = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        sub = [rows[i] for i in idx]
        y = [r["changed"] for r in sub]
        pred = [r["pred"] for r in sub]
        logit = [r["logit"] for r in sub]
        bal.append(balanced_acc(y, pred))
        aucs.append(auc(y, logit))
        struct.append(structure_summary(sub, train_mean=train_mean))
    point = structure_summary(rows, train_mean=train_mean)
    ci_keys = (
        "r2_mean_all",
        "r2_median_all",
        "r2_mean_nondegenerate",
        "skill_mean",
        "spearman_mean",
        "spearman_mean_nondegenerate",
        "mae_ratio_median",
        "presence_auc_mean",
    )
    return {
        "n": len(rows),
        "n_groups": int(len(uniq)),
        "balanced": balanced_acc(
            [r["changed"] for r in rows], [r["pred"] for r in rows]
        ),
        "balanced_ci": np.nanpercentile(bal, [2.5, 97.5]).tolist(),
        "auc": auc([r["changed"] for r in rows], [r["logit"] for r in rows]),
        "auc_ci": np.nanpercentile(aucs, [2.5, 97.5]).tolist(),
        "structure": point,
        "structure_ci": {k: _nan_ci(struct, k) for k in ci_keys},
    }


def paired_bootstrap(real, blank, n_boot=5000, seed=0):
    b = {r["ex_id"]: r for r in blank}
    pairs = [(r, b[r["ex_id"]]) for r in real if r["ex_id"] in b]
    if not pairs:
        return {}
    groups = np.asarray([r["patient_id"] for r, _ in pairs])
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    a = np.asarray([r["correct"] for r, _ in pairs], dtype=float)
    bb = np.asarray([q["correct"] for _, q in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        deltas.append(float(a[idx].mean() - bb[idx].mean()))
    return {
        "n": len(pairs),
        "n_groups": int(len(uniq)),
        "delta": float(a.mean() - bb.mean()),
        "ci": np.percentile(deltas, [2.5, 97.5]).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--blank", default=None)
    ap.add_argument("--train-data", dest="train_data", default=None)
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from pilot.data import read_jsonl
    from pilot.structure_eval import log1p_target

    train_mean = None
    if args.train_data:
        train = read_jsonl(args.train_data)
        if not train:
            raise SystemExit("empty training data")
        # The prediction width is defined by the first real row's s_pred.
        first = read_rows(args.real)[0]
        d = len(first["s_pred"])
        train_mean = np.nanmean(log1p_target([r["descriptor"] for r in train], d), axis=0)

    real = read_rows(args.real)
    result = {
        "real": group_bootstrap(real, train_mean, args.n_boot),
    }
    print("real:", json.dumps(result["real"], indent=2, default=float))
    if args.blank:
        blank = read_rows(args.blank)
        result["blank"] = group_bootstrap(blank, train_mean, args.n_boot)
        result["real_minus_blank"] = paired_bootstrap(real, blank, args.n_boot)
        print("blank:", json.dumps(result["blank"], indent=2, default=float))
        print(
            "real-blank paired:",
            json.dumps(result["real_minus_blank"], indent=2, default=float),
        )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
