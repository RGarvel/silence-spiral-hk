"""Quantities used to compare runs: clustering, polarization, convergence."""

from __future__ import annotations

import numpy as np


def cluster_opinions(opinions: np.ndarray, eps: float) -> list[np.ndarray]:
    """Gap-based clustering of final opinions (standard for BC models):
    sort opinions, start a new cluster wherever the gap exceeds eps."""
    o = np.sort(opinions)
    if o.size == 0:
        return []
    splits = np.flatnonzero(np.diff(o) > eps) + 1
    return [c for c in np.split(o, splits)]


def n_clusters(opinions: np.ndarray, eps: float) -> int:
    return len(cluster_opinions(opinions, eps))


def largest_cluster_share(opinions: np.ndarray, eps: float) -> float:
    """Share of agents in the largest opinion cluster — a dominance
    measure aligned with the paper's 'dominated opinions are formed'."""
    clusters = cluster_opinions(opinions, eps)
    if not clusters:
        return 0.0
    return max(len(c) for c in clusters) / len(opinions)


def opinion_spread(opinions: np.ndarray) -> float:
    """Standard deviation of opinions (dispersion / weak polarization)."""
    return float(np.std(opinions))


def convergence_time(max_delta: list[float], tol: float = 1e-4) -> int:
    """First step at which the maximal opinion change drops below `tol`;
    len(max_delta) if never reached."""
    for t, d in enumerate(max_delta, start=1):
        if d < tol:
            return t
    return len(max_delta)


def silenced_edge_fraction(weights: np.ndarray, threshold: float = 0.1) -> float:
    """Fraction of arcs whose influence weight fell below `threshold` —
    an operational 'silenced voices' indicator for the SOS mechanism."""
    if weights.size == 0:
        return 0.0
    return float(np.mean(weights < threshold))
