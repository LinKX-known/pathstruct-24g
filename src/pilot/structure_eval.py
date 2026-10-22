"""Honest evaluation of continuous structure targets.

``R^2`` is a bad headline for the topology block.  ``hole_area_frac`` is zero on
86-97% of held-out patches and has a variance around ``1e-6`` in log1p space;
one bad prediction on a rare, large lumen can therefore make the R^2 of that
single dimension −30 to −200 and dominate an unweighted mean over five
dimensions.  The negative number is real (the model *is* bad on that
dimension), but it says nothing about the other four, and averaging it with
them produces a meaningless "the structure head failed" conclusion.

This module keeps the R^2 but always reports it next to:

* a **constant train-mean predictor** on the same held-out rows (the baseline
  R^2 that any model has to beat);
* the **skill** relative to that predictor (``1 - SSE_model / SSE_mean``);
* **log-space MAE**, normalised by the target's spread and by the mean
  predictor's MAE;
* **Spearman rho**, which is invariant to monotone scale errors;
* **calibration** (mean prediction minus mean truth);
* separate **presence / positive-value** metrics for zero-inflated dimensions
  (AUC for "is there a lumen?", MAE/R^2 on the patches that have one).

A dimension is flagged ``degenerate`` when its held-out variance is tiny
(``< 1e-5``) or when at least 80% of values are zero.  Aggregates over all
dimensions are still reported for continuity with the old report, but the
headline numbers are the median/non-degenerate ones.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

EPS = 1e-12


def log1p_target(values: Any, dims: Optional[int] = None) -> np.ndarray:
    """The space the structure head is trained in (non-negative log1p)."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if dims is not None:
        a = a[:, :dims]
    return np.log1p(np.clip(a, 0.0, None))


def _finite_pair(true: np.ndarray, pred: np.ndarray):
    ok = np.isfinite(true) & np.isfinite(pred)
    return true[ok], pred[ok], ok


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred, _ = _finite_pair(true, pred)
    if len(true) < 2:
        return float("nan")
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot <= EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _spearman(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred, _ = _finite_pair(true, pred)
    if len(true) < 3 or np.std(true) <= EPS or np.std(pred) <= EPS:
        return float("nan")
    from scipy.stats import spearmanr

    rho = spearmanr(true, pred).correlation
    return float(rho) if np.isfinite(rho) else float("nan")


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _balanced_binary(labels: np.ndarray, pred: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    vals = []
    for target in (False, True):
        mask = labels == target
        if mask.any():
            vals.append(float(np.mean(pred[mask] == target)))
    return float(np.mean(vals)) if vals else float("nan")


def per_dim_diagnostics(
    true: np.ndarray,
    pred: np.ndarray,
    names: Optional[Sequence[str]] = None,
    train_mean: Optional[np.ndarray] = None,
    *,
    degenerate_var: float = 1e-5,
    degenerate_zero_frac: float = 0.80,
) -> List[Dict[str, Any]]:
    """Per-dimension diagnostics for targets already in log1p space.

    ``train_mean`` must be a constant-predictor baseline computed on the
    *training* split (not the held-out split).  Using the held-out mean would
    make the baseline trivially 0 and hide the failure mode this module exists
    to catch.
    """
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if true.ndim == 1:
        true = true[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]
    if true.shape != pred.shape:
        raise ValueError(f"true {true.shape} and pred {pred.shape} must match")
    d = true.shape[1]
    if names is None:
        names = [f"dim{j}" for j in range(d)]
    names = list(names)[:d]
    if train_mean is not None:
        train_mean = np.asarray(train_mean, dtype=np.float64).reshape(-1)
        if len(train_mean) < d:
            raise ValueError("train_mean must have one value per dimension")

    out: List[Dict[str, Any]] = []
    for j in range(d):
        t, p, ok = _finite_pair(true[:, j], pred[:, j])
        n = int(ok.sum())
        row: Dict[str, Any] = {"name": names[j], "index": j, "n": n}
        if n == 0:
            row.update(
                {
                    "degenerate": True,
                    "degen_reason": "no finite pairs",
                    "r2": float("nan"),
                }
            )
            out.append(row)
            continue

        mu_t = float(np.mean(t))
        mu_p = float(np.mean(p))
        var_t = float(np.var(t))
        std_t = float(np.sqrt(var_t))
        ss_res = float(np.sum((p - t) ** 2))
        ss_tot = float(np.sum((t - mu_t) ** 2))
        mae = float(np.mean(np.abs(p - t)))
        zero_frac = float(np.mean(t == 0.0))

        row.update(
            {
                "n": n,
                "mean_true": mu_t,
                "mean_pred": mu_p,
                "std_true": std_t,
                "std_pred": float(np.std(p)),
                "var_true": var_t,
                "zero_frac": zero_frac,
                "mse": float(np.mean((p - t) ** 2)),
                "mae": mae,
                "mae_norm": mae / max(std_t, EPS),
                "mae_over_std": mae / max(std_t, EPS),
                "calibration_gap": mu_p - mu_t,
                "r2": _r2(t, p),
                "r2_test_mean": 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan"),
                "spearman": _spearman(t, p),
            }
        )

        if train_mean is not None:
            tm = float(train_mean[j])
            ss_mean = float(np.sum((t - tm) ** 2))
            mae_mean = float(np.mean(np.abs(t - tm)))
            row.update(
                {
                    "train_mean": tm,
                    "r2_train_mean": 1.0 - ss_mean / ss_tot if ss_tot > EPS else float("nan"),
                    "skill_vs_train_mean": 1.0 - ss_res / ss_mean if ss_mean > EPS else float("nan"),
                    "mae_train_mean": mae_mean,
                    "mae_ratio_vs_train_mean": mae / max(mae_mean, EPS),
                }
            )

        degenerate = var_t <= degenerate_var or zero_frac >= degenerate_zero_frac
        row["degenerate"] = bool(degenerate)
        if degenerate:
            row["degen_reason"] = (
                "near-zero variance" if var_t <= degenerate_var else "zero-inflated"
            )

        # Zero-inflated dimensions need a presence metric; a continuous R^2
        # mixes "did you find the rare lumen?" with "did you get its size
        # right?" and is dominated by the first question.
        present = t > 0.0
        row["n_positive"] = int(present.sum())
        if present.sum() >= 3:
            t_pos, p_pos = t[present], p[present]
            row["mae_positive"] = float(np.mean(np.abs(p_pos - t_pos)))
            row["r2_positive"] = _r2(t_pos, p_pos)
            row["spearman_positive"] = _spearman(t_pos, p_pos)
            row["mean_true_positive"] = float(np.mean(t_pos))
            row["mean_pred_positive"] = float(np.mean(p_pos))
        else:
            row["mae_positive"] = float("nan")
            row["r2_positive"] = float("nan")
            row["spearman_positive"] = float("nan")
        row["presence_auc"] = _auc(present, p)
        row["presence_balanced_acc"] = _balanced_binary(present, p > 0.0)
        out.append(row)
    return out


def summary(diags: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Robust aggregate over per-dimension diagnostics.

    ``r2_mean_all`` is kept for comparability with the old report.  Prefer
    ``r2_median_all``, ``r2_mean_nondegenerate``, ``skill_mean`` and
    ``spearman_mean`` when the block contains zero-inflated dimensions.
    """
    diags = list(diags)
    if not diags:
        return {"n_dims": 0}

    def vals(key: str, subset: Optional[Iterable[Dict[str, Any]]] = None):
        src = list(subset) if subset is not None else diags
        return np.asarray([d.get(key, np.nan) for d in src], dtype=np.float64)

    def nanmean(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

    def nanmedian(a: np.ndarray) -> float:
        return float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")

    nondeg = [d for d in diags if not d.get("degenerate", False)]
    zero_inflated = [d for d in diags if d.get("zero_frac", 0.0) > 0.0]
    r2_all = vals("r2")
    skill = vals("skill_vs_train_mean")
    mae_ratio = vals("mae_ratio_vs_train_mean")
    rho = vals("spearman")
    return {
        "n_dims": len(diags),
        "n_degenerate": sum(bool(d.get("degenerate", False)) for d in diags),
        "r2_mean_all": nanmean(r2_all),
        "r2_median_all": nanmedian(r2_all),
        "r2_mean_nondegenerate": nanmean(vals("r2", nondeg)),
        "r2_train_mean_mean": nanmean(vals("r2_train_mean")),
        "skill_mean": nanmean(skill),
        "skill_median": nanmedian(skill),
        "mae_ratio_mean": nanmean(mae_ratio),
        "mae_ratio_median": nanmedian(mae_ratio),
        "mae_norm_mean": nanmean(vals("mae_norm")),
        "spearman_mean": nanmean(rho),
        "spearman_median": nanmedian(rho),
        "spearman_mean_nondegenerate": nanmean(vals("spearman", nondeg)),
        "mae_ratio_median_nondegenerate": nanmedian(vals("mae_ratio_vs_train_mean", nondeg)),
        "calibration_gap_abs_mean": nanmean(np.abs(vals("calibration_gap"))),
        "presence_auc_mean": nanmean(vals("presence_auc", zero_inflated)),
        "presence_balanced_mean": nanmean(vals("presence_balanced_acc", zero_inflated)),
        "mae_positive_mean": nanmean(vals("mae_positive", zero_inflated)),
        "r2_positive_mean": nanmean(vals("r2_positive", zero_inflated)),
        "spearman_positive_mean": nanmean(vals("spearman_positive", zero_inflated)),
    }


def format_table(diags: Sequence[Dict[str, Any]]) -> str:
    """Fixed-width table for a terminal; one row per descriptor dimension."""
    header = (
        f"{'dim':18s} {'zero%':>6s} {'var':>10s} {'mean_t':>9s} {'mean_p':>9s} "
        f"{'R2':>8s} {'R2mean':>8s} {'skill':>8s} {'MAE':>8s} {'MAE/mean':>9s} "
        f"{'rho':>7s} {'calib':>8s}  flag"
    )
    lines = [header, "-" * len(header)]
    for d in diags:
        def f(key: str, fmt: str = ".3f") -> str:
            v = d.get(key, float("nan"))
            try:
                if not np.isfinite(v):
                    return "nan"
                return format(v, fmt)
            except (TypeError, ValueError):
                return "nan"

        flag = d.get("degen_reason", "") if d.get("degenerate") else ""
        lines.append(
            f"{str(d.get('name', '?'))[:18]:18s} "
            f"{100.0 * d.get('zero_frac', 0.0):6.1f} "
            f"{d.get('var_true', float('nan')):10.3g} "
            f"{f('mean_true', '.4f'):>9s} {f('mean_pred', '.4f'):>9s} "
            f"{f('r2', '.3f'):>8s} {f('r2_train_mean', '.3f'):>8s} "
            f"{f('skill_vs_train_mean', '.3f'):>8s} {f('mae', '.4f'):>8s} "
            f"{f('mae_ratio_vs_train_mean', '.2f'):>9s} {f('spearman', '.3f'):>7s} "
            f"{f('calibration_gap', '+.4f'):>8s}  {flag}"
        )
    return "\n".join(lines)


def diagnostics_payload(
    diagnostics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """JSON-serialisable bundle used by the eval scripts."""
    return {"per_dim": list(diagnostics), "summary": summary(diagnostics)}
