#!/usr/bin/env python
"""Compare hurdle and plain structure heads on zero-inflated dimensions.

For each ``heads_*.json`` written by ``eval_heads.py`` this reports, per
zero-inflated dimension:

* presence AUC / balanced accuracy from the dedicated presence logit when the
  run has one (``s_presence``), otherwise from ``s_pred > 0``;
* positive-only MAE, MAE relative to the training-mean positive predictor,
  R^2 and Spearman using the conditional regression output (``s_reg``) when
  available, otherwise the plain prediction;
* the usual expected-value R^2 and MAE ratio from ``structure_eval``.

The training JSONL must be the same one used to train the adapter so that the
mean predictors are computed on the training split.

    python scripts/eval/diagnose_hurdle.py \
        --pred out/heads_topo_real_s0.json \
        --pred out/heads_hurdle_real_s0.json \
        --train-data data/real/monuseg_train/cf_train_bal.jsonl \
        --dims 5 --out out/hurdle_compare_monuseg.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pilot.structure_eval import log1p_target, per_dim_diagnostics  # noqa: E402

TOPO_KEYS = ["n_comp", "hole_count", "hole_area_frac", "h0_max", "h0_sum"]


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def dim_names(d: int):
    return TOPO_KEYS[:d] + [f"dim{i}" for i in range(len(TOPO_KEYS), d)]


def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _r2(true, pred):
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def _positive_metrics(true_log, pred_log, train_positive_mean):
    """MAE/R2/Spearman on rows where the true raw value is positive."""
    mae = float(np.mean(np.abs(pred_log - true_log)))
    base_mae = float(np.mean(np.abs(train_positive_mean - true_log)))
    return {
        "n_pos": int(len(true_log)),
        "mae": mae,
        "mae_ratio": mae / base_mae if base_mae > 0 else float("nan"),
        "r2": _r2(true_log, pred_log),
        "spearman": _spearman(true_log, pred_log),
    }


def diagnose_run(pred_path, train_rows, dims):
    rows = read_rows(pred_path)
    true_raw = np.asarray([r["descriptor"][:dims] for r in rows], dtype=np.float64)
    true_log = log1p_target(true_raw)
    pred_log = np.asarray([r["s_pred"][:dims] for r in rows], dtype=np.float64)
    train_raw = np.asarray([r["descriptor"][:dims] for r in train_rows], dtype=np.float64)
    train_log = log1p_target(train_raw)
    train_mean = np.nanmean(train_log, axis=0)

    diags = per_dim_diagnostics(
        true_log, pred_log, names=dim_names(dims), train_mean=train_mean
    )
    hurdle_dims = [int(i) for i in rows[0].get("hurdle_dims", [])]
    out_rows = []
    for i in range(dims):
        d = diags[i]
        present = true_raw[:, i] > 0
        if hurdle_dims:
            j = hurdle_dims.index(i) if i in hurdle_dims else None
        else:
            j = None
        if j is not None and "s_presence" in rows[0]:
            score = np.asarray([r["s_presence"][j] for r in rows], dtype=float)
            reg = np.asarray([r["s_reg"][i] for r in rows], dtype=float)
            presence_auc = (
                float(roc_auc_score(present, score)) if len(set(present.tolist())) > 1 else float("nan")
            )
            presence_bal = (
                float(balanced_accuracy_score(present, score >= 0.5))
                if len(set(present.tolist())) > 1
                else float("nan")
            )
        else:
            score = pred_log[:, i]
            reg = pred_log[:, i]
            presence_auc = d.get("presence_auc", float("nan"))
            presence_bal = d.get("presence_balanced_acc", float("nan"))
        if d.get("zero_frac", 0.0) < 0.5:
            # Presence is not a meaningful question for a dense dimension.
            presence_auc = float("nan")
            presence_bal = float("nan")
        pos = present & np.isfinite(true_log[:, i]) & np.isfinite(reg)
        if pos.any():
            train_pos = train_raw[:, i] > 0
            base = float(np.mean(train_log[train_pos, i])) if train_pos.any() else float("nan")
            pm = _positive_metrics(true_log[pos, i], reg[pos], base)
        else:
            pm = {"n_pos": 0, "mae": float("nan"), "mae_ratio": float("nan"),
                  "r2": float("nan"), "spearman": float("nan")}
        out_rows.append(
            {
                "dim": d["name"],
                "index": i,
                "zero_frac": d.get("zero_frac", float("nan")),
                "presence_auc": presence_auc,
                "presence_balanced": presence_bal,
                **{f"positive_{k}": v for k, v in pm.items()},
                "expected_r2": d.get("r2", float("nan")),
                "expected_mae_ratio": d.get("mae_ratio_vs_train_mean", float("nan")),
                "expected_spearman": d.get("spearman", float("nan")),
            }
        )
    return {
        "pred": str(pred_path),
        "n": len(rows),
        "hurdle_dims": hurdle_dims,
        "per_dim": out_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--train-data", action="append", required=True)
    ap.add_argument("--dims", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if len(args.train_data) == 1:
        train_paths = args.train_data * len(args.pred)
    elif len(args.train_data) == len(args.pred):
        train_paths = args.train_data
    else:
        raise SystemExit("pass one --train-data or one per --pred")
    payload = {"dims": args.dims, "runs": {}}
    for pred_path, train_path in zip(args.pred, train_paths):
        result = diagnose_run(pred_path, read_rows(train_path), args.dims)
        payload["runs"][str(pred_path)] = result
        print(f"\n=== {pred_path} | hurdle_dims={result['hurdle_dims']} ===")
        print(
            "dim              zero%  presAUC  presBal  n_pos   posMAE  posMAE/mean  posR2   posRho  expR2   expMAE/mean"
        )
        for r in result["per_dim"]:
            print(
                f"{r['dim']:16s} {100*r['zero_frac']:5.0f}  "
                f"{r['presence_auc']:7.3f}  {r['presence_balanced']:7.3f}  "
                f"{r['positive_n_pos']:5d}  {r['positive_mae']:7.3f}  "
                f"{r['positive_mae_ratio']:11.2f}  {r['positive_r2']:7.3f}  "
                f"{r['positive_spearman']:7.3f}  {r['expected_r2']:7.3f}  "
                f"{r['expected_mae_ratio']:11.2f}"
            )
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
