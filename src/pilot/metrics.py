"""Metrics that check observations, not just answers.

The probe's d' answers "can the model tell two arrangements apart". These
answer "when the model is right, is it right for the stated reason".
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def within_tolerance(pred: np.ndarray, true: np.ndarray, rel: float = 0.1, abs_min: float = 1.0) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    tol = np.maximum(abs_min, rel * np.abs(true))
    return np.abs(pred - true) <= tol


def rawr_struct(
    pred: np.ndarray,
    true: np.ndarray,
    correct: Sequence[bool],
    rel: float = 0.1,
    abs_min: float = 1.0,
) -> float:
    """Right answer, wrong structure: fraction of out-of-tolerance descriptors
    among the answers that were correct. Lower is better; NaN if none correct."""
    correct = np.asarray(correct, dtype=bool)
    if correct.sum() == 0:
        return float("nan")
    ok = within_tolerance(np.asarray(pred)[correct], np.asarray(true)[correct], rel, abs_min)
    if ok.ndim == 1:
        return float(1.0 - ok.mean())
    return float(1.0 - ok.mean(axis=1).mean())


def coherent_accuracy(
    pred: np.ndarray,
    rule,
    expected: Sequence[object],
) -> float:
    """Fraction of responses whose *reported descriptors* imply the right answer.

    ``rule`` maps one descriptor row to an answer; it is the deterministic
    decision rule of the task (e.g. "more than 3 closed lumina -> acinar").
    ``expected`` is the reference answer per row.
    """
    pred = np.asarray(pred, dtype=np.float64)
    if len(pred) == 0:
        return float("nan")
    implied = [rule(row) for row in pred]
    hits = [a == b for a, b in zip(implied, expected)]
    return float(np.mean(hits))


def structure_change_accuracy(pred_changed: Sequence[int], true_changed: Sequence[int]) -> Dict[str, float]:
    p = np.asarray(pred_changed, dtype=int)
    t = np.asarray(true_changed, dtype=int)
    if len(p) == 0:
        return {"n": 0}
    acc = float((p == t).mean())
    pos = float((p[t == 1] == 1).mean()) if (t == 1).any() else float("nan")
    neg = float((p[t == 0] == 0).mean()) if (t == 0).any() else float("nan")
    return {
        "n": int(len(p)),
        "accuracy": acc,
        "sensitivity": pos,
        "specificity": neg,
        "changed_rate_true": float(t.mean()),
        "changed_rate_pred": float(p.mean()),
    }


def class_balance(labels: Sequence[int]) -> Dict[str, float]:
    y = np.asarray(labels, dtype=int)
    if len(y) == 0:
        return {"n": 0}
    return {"n": int(len(y)), "positive_rate": float(y.mean())}
