"""Inertial Hegselmann-Krause model (Chazelle & Wang, IEEE TAC 2017).

Agents do not jump fully to the mean of their confidence neighbors; a
per-agent inertia lambda trades off own opinion against the neighbors':

    O_i(t+1) = (1 - lam) * O_i(t) + lam * mean_{j in N_i(t)} O_j(t)

lam = 1 degenerates to classical HK; lam = 0 is a stubborn ("close-minded")
agent. In the UV 2022 paper the compared IHK used self-confidence 0.7,
i.e. lam = 0.3.
"""

from __future__ import annotations

import numpy as np

from spiral_hk.graph import Graph
from spiral_hk.models.hegselmann_krause import confidence_mask


def ihk_step(
    opinions: np.ndarray, g: Graph, eps: float, inertia: float = 0.3
) -> np.ndarray:
    """One synchronous IHK update. `inertia` is the weight lam on the
    neighbors' mean (lam = 1 - self-confidence)."""
    if not 0.0 <= inertia <= 1.0:
        raise ValueError("inertia must be in [0, 1]")
    mask = confidence_mask(opinions, g, eps)
    src, dst = g.src[mask], g.dst[mask]
    count = np.bincount(src, minlength=g.n)
    total = np.bincount(src, weights=opinions[dst], minlength=g.n)
    new_opinions = opinions.copy()
    active = count > 0
    neighbor_mean = total[active] / count[active]
    new_opinions[active] = (1.0 - inertia) * opinions[active] + inertia * neighbor_mean
    return new_opinions
