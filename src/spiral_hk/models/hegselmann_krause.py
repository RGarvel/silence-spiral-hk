"""Classical Hegselmann-Krause bounded-confidence model on graphs.

Update rule (Hegselmann & Krause, JASSS 2002):

    O_i(t+1) = mean_{j in N_i(t)} O_j(t),   N_i(t) = { j ~ i : |O_i - O_j| <= eps }

Agents with no confidence-bounded neighbor keep their opinion.
"""

from __future__ import annotations

import numpy as np

from spiral_hk.graph import Graph


def confidence_mask(opinions: np.ndarray, g: Graph, eps: float) -> np.ndarray:
    """Boolean mask over arcs: True iff the speaker is within the
    listener's confidence bound (|O_src - O_dst| <= eps)."""
    return np.abs(opinions[g.src] - opinions[g.dst]) <= eps


def hk_step(opinions: np.ndarray, g: Graph, eps: float) -> np.ndarray:
    """One synchronous HK update. Returns a new opinion vector."""
    mask = confidence_mask(opinions, g, eps)
    src, dst = g.src[mask], g.dst[mask]
    count = np.bincount(src, minlength=g.n)
    total = np.bincount(src, weights=opinions[dst], minlength=g.n)
    new_opinions = opinions.copy()
    active = count > 0
    new_opinions[active] = total[active] / count[active]
    return new_opinions
