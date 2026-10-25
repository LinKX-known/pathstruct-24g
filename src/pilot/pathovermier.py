"""ASPECT's PathoVernier metrics, implemented as defined in their paper.

To be comparable with ASPECT the table has to use *their* numbers, not ours:

``S(C, Q)``  fraction of required counts that fall within tolerance
             ``tau_i = max(1, 0.1 * q_i)``
``CA``       coherent accuracy: does the count-derived decision agree with the
             answer, given that a decision is derivable at all
``RAWR``     right answer, wrong reason: ``1 - S`` among correct answers whose
             required counts were all reported

Our structure observation is an addition, not a replacement: the model must still
report cell counts, otherwise CA and RAWR cannot be computed and the comparison
has holes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np


def within_tolerance(pred: float, true: float, rel: float = 0.1, abs_min: float = 1.0) -> bool:
    tau = max(abs_min, rel * abs(true))
    return abs(pred - true) <= tau


def count_score(
    reported: Mapping[str, float],
    required: Mapping[str, float],
    rel: float = 0.1,
    abs_min: float = 1.0,
) -> float:
    """``S(C, Q)``: every required count contributes equally, missing ones score 0."""
    if not required:
        return float("nan")
    hits = 0
    for key, value in required.items():
        if key in reported:
            hits += int(within_tolerance(float(reported[key]), float(value), rel, abs_min))
    return hits / len(required)


def coherent_accuracy(
    decisions: Sequence[Optional[Any]],
    expected: Sequence[Any],
) -> float:
    """``CA``: agreement between the count-derived decision and the answer.

    ``decisions`` may contain ``None`` where the rule is not defined; those rows
    are excluded from the denominator, exactly as in the paper.
    """
    pairs = [(d, e) for d, e in zip(decisions, expected) if d is not None]
    if not pairs:
        return float("nan")
    return float(np.mean([d == e for d, e in pairs]))


def rawr(
    s_values: Sequence[float],
    answer_correct: Sequence[bool],
    complete_counts: Sequence[bool],
) -> float:
    """``RAWR``: among correct answers with all required counts reported, ``1 - S``."""
    mask = [
        i for i, (ok, complete) in enumerate(zip(answer_correct, complete_counts))
        if ok and complete
    ]
    if not mask:
        return float("nan")
    s = np.asarray([s_values[i] for i in mask], dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(1.0 - s.mean())


def summarise(
    reported: Sequence[Mapping[str, float]],
    required: Sequence[Mapping[str, float]],
    answer_correct: Sequence[bool],
    decisions: Sequence[Optional[Any]],
    expected: Sequence[Any],
    complete_counts: Sequence[bool],
) -> Dict[str, float]:
    """The three numbers that go in the comparison table."""
    s = [count_score(r, q) for r, q in zip(reported, required)]
    return {
        "n": len(s),
        "S": float(np.nanmean(s)) if s else float("nan"),
        "CA": coherent_accuracy(decisions, expected),
        "RAWR": rawr(s, answer_correct, complete_counts),
        "answer_accuracy": float(np.mean(answer_correct)) if answer_correct else float("nan"),
    }
