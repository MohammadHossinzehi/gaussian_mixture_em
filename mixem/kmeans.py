"""k-means++ seeding and Lloyd's algorithm.

Used to initialise the mixture models, but also usable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape (n_samples, n_centers)."""
    d = (
        np.sum(X * X, axis=1)[:, None]
        - 2.0 * X @ C.T
        + np.sum(C * C, axis=1)[None, :]
    )
    return np.maximum(d, 0.0)


def kmeans_plusplus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Arthur and Vassilvitskii (2007) D^2 seeding.

    Each new center is drawn with probability proportional to its squared
    distance from the nearest center chosen so far, which gives an
    O(log k) competitive guarantee in expectation.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, n_samples], got {k}")
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(n)]
    closest = _sq_dists(X, centers[:1])[:, 0]
    for j in range(1, k):
        total = closest.sum()
        if total <= 0.0:
            # every point coincides with a center: pick uniformly
            idx = rng.integers(n)
        else:
            idx = rng.choice(n, p=closest / total)
        centers[j] = X[idx]
        closest = np.minimum(closest, _sq_dists(X, centers[j : j + 1])[:, 0])
    return centers


@dataclass
class KMeansResult:
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    history: list


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-10,
    seed: int | None = None,
) -> KMeansResult:
    """Lloyd's algorithm from k-means++ seeds; best of ``n_init`` restarts.

    ``history`` holds the inertia after every assignment step of the
    winning run; it is non increasing by construction, which the test
    suite checks.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    best: KMeansResult | None = None
    for _ in range(n_init):
        centers = kmeans_plusplus(X, k, rng)
        history = []
        labels = None
        for it in range(1, max_iter + 1):
            d = _sq_dists(X, centers)
            labels = np.argmin(d, axis=1)
            inertia = float(d[np.arange(len(X)), labels].sum())
            history.append(inertia)
            new_centers = centers.copy()
            for j in range(k):
                members = X[labels == j]
                if len(members):
                    new_centers[j] = members.mean(axis=0)
                else:
                    # re-seed an empty cluster at the worst fitted point
                    far = int(np.argmax(d[np.arange(len(X)), labels]))
                    new_centers[j] = X[far]
            shift = float(np.sum((new_centers - centers) ** 2))
            centers = new_centers
            if shift <= tol:
                break
        d = _sq_dists(X, centers)
        labels = np.argmin(d, axis=1)
        inertia = float(d[np.arange(len(X)), labels].sum())
        run = KMeansResult(centers, labels, inertia, it, history)
        if best is None or run.inertia < best.inertia:
            best = run
    assert best is not None
    return best
