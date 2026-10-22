#!/usr/bin/env python
"""Patient/tissue-grouped bootstrap for the trained-head evaluation.

Consumes the JSON arrays written by ``scripts/eval_heads.py --out`` (real and,
optionally, blank-box control) and reports:

* balanced accuracy / AUC with a 95% group bootstrap;
* real-minus-blank paired difference (same ex_id);
* structure-head topology R^2 with a group bootstrap.

    python scripts/bootstrap_heads.py \
        --real out/heads_real_s0.json --blank out/heads_blank_s0.json
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


def r2_topology(rows):
    true = np.asarray([r["descriptor"] for r in rows], dtype=float)
    pred = np.asarray([r["s_pred"] for r in rows], dtype=float)
    d = min(pred.shape[1], true.shape[1])
    target = np.log1p(np.clip(true[:, :d], 0, None))
    pred = pred[:, :d]
    out = []
    for j in range(d):
        t = target[:, j]
        p = pred[:, j]
        ss_res = float(np.sum((p - t) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        out.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
    return float(np.nanmean(out[:5])), float(np.nanmean(out))


def group_bootstrap(rows, n_boot=5000, seed=0):
    groups = np.asarray([r["patient_id"] for r in rows])
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    bal, aucs, topo = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        sub = [rows[i] for i in idx]
        y = [r["changed"] for r in sub]
        pred = [r["pred"] for r in sub]
        logit = [r["logit"] for r in sub]
        bal.append(balanced_acc(y, pred))
        aucs.append(auc(y, logit))
        topo.append(r2_topology(sub)[0])
    return {
        "n": len(rows),
        "n_groups": int(len(uniq)),
        "balanced": balanced_acc(
            [r["changed"] for r in rows], [r["pred"] for r in rows]
        ),
        "balanced_ci": np.nanpercentile(bal, [2.5, 97.5]).tolist(),
        "auc": auc([r["changed"] for r in rows], [r["logit"] for r in rows]),
        "auc_ci": np.nanpercentile(aucs, [2.5, 97.5]).tolist(),
        "r2_topology": r2_topology(rows)[0],
        "r2_topology_ci": np.nanpercentile(topo, [2.5, 97.5]).tolist(),
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
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=5000)
    args = ap.parse_args()

    real = read_rows(args.real)
    print("real:", json.dumps(group_bootstrap(real, args.n_boot), indent=2))
    if args.blank:
        blank = read_rows(args.blank)
        print("blank:", json.dumps(group_bootstrap(blank, args.n_boot), indent=2))
        print("real-blank paired:", json.dumps(paired_bootstrap(real, blank, args.n_boot), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
