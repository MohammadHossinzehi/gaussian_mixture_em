"""Small numerical helpers: log-sum-exp, digamma and log-gamma.

Kept dependency free (NumPy only) so the package does not need SciPy.
"""

from __future__ import annotations

import math

import numpy as np

_EULER_GAMMA = 0.5772156649015329


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable log(sum(exp(a))) along ``axis``."""
    a = np.asarray(a, dtype=float)
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


def _digamma_scalar(x: float) -> float:
    if x <= 0 and float(x).is_integer():
        return float("nan")
    result = 0.0
    if x < 0:
        # reflection: psi(1 - x) - psi(x) = pi * cot(pi * x)
        return _digamma_scalar(1.0 - x) - math.pi / math.tan(math.pi * x)
    # recurrence psi(x) = psi(x + 1) - 1/x until x is large enough
    while x < 10.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    series = inv2 * (
        1.0 / 12
        - inv2 * (1.0 / 120 - inv2 * (1.0 / 252 - inv2 * (1.0 / 240 - inv2 * (1.0 / 132))))
    )
    return result + math.log(x) - 0.5 * inv - series


def digamma(x):
    """Digamma function psi(x) = d/dx log Gamma(x), elementwise."""
    arr = np.asarray(x, dtype=float)
    out = np.vectorize(_digamma_scalar, otypes=[float])(arr)
    return out if out.shape else float(out)


def gammaln(x):
    """log|Gamma(x)| elementwise via math.lgamma."""
    arr = np.asarray(x, dtype=float)
    out = np.vectorize(math.lgamma, otypes=[float])(arr)
    return out if out.shape else float(out)


EULER_GAMMA = _EULER_GAMMA
