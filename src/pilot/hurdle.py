"""Hurdle decomposition for zero-inflated structure targets.

``hole_count`` and ``hole_area_frac`` are zero for 87-96% of the real-data
patches.  A single smooth-L1 regression on ``log1p(x)`` is dominated by the
zero spike: the head can minimise the loss by predicting a small value for
every row, and the positive values are never fitted.  This module provides the
small amount of shared logic needed to:

* pick the zero-inflated dimensions from the *training* descriptors;
* give each one a positive-class weight for the presence BCE;
* turn the regression output into a log1p-space point estimate
  ``P(x > 0) * E[log1p(x) | x > 0]`` for the existing diagnostics.

The head itself outputs ``dims + len(hurdle_dims)`` values: the first ``dims``
are the regression outputs (all in log1p space; zero-inflated dims are fitted
only on positive rows), the remaining ones are presence logits.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

MAX_POS_WEIGHT = 20.0


def zero_inflated_dims(
    descriptors: Iterable[Sequence[float]],
    dims: int,
    threshold: float = 0.5,
) -> Tuple[List[int], List[float]]:
    """Return ``(indices, zero_fractions)`` for dims with zero_frac >= threshold."""
    rows = [list(d[:dims]) for d in descriptors if d is not None and len(d) >= dims]
    if not rows:
        return [], [0.0] * dims
    arr = np.asarray(rows, dtype=np.float64)
    zero_frac = np.mean(arr <= 0.0, axis=0)
    idx = [i for i in range(dims) if float(zero_frac[i]) >= threshold]
    return idx, [float(v) for v in zero_frac]


def positive_class_weights(
    descriptors: Iterable[Sequence[float]],
    hurdle_dims: Sequence[int],
) -> List[float]:
    """``neg/pos`` per hurdle dimension, clipped to [1, MAX_POS_WEIGHT]."""
    weights: List[float] = []
    rows = [d for d in descriptors if d is not None]
    if not rows or not hurdle_dims:
        return weights
    arr = np.asarray(rows, dtype=np.float64)
    for dim in hurdle_dims:
        pos = float(np.mean(arr[:, dim] > 0.0))
        if pos <= 0.0:
            weights.append(1.0)
        else:
            weights.append(float(min(MAX_POS_WEIGHT, max(1.0, (1.0 - pos) / pos))))
    return weights


def expected_log1p(
    regression: Sequence[float],
    presence: Sequence[float] | None,
    hurdle_dims: Sequence[int],
) -> np.ndarray:
    """Combine the two parts into a log1p-space point estimate.

    ``regression`` is the head's ``dims`` regression outputs; ``presence`` are
    the sigmoid probabilities aligned with ``hurdle_dims``.  For a hurdle
    dimension the conditional mean is multiplied by the presence probability:
    ``E[log1p(x)] = P(x > 0) * E[log1p(x) | x > 0]``.
    """
    out = np.asarray(regression, dtype=np.float64).copy()
    if presence is None:
        return out
    for j, dim in enumerate(hurdle_dims):
        if dim < len(out):
            out[dim] = float(presence[j]) * float(out[dim])
    return out
