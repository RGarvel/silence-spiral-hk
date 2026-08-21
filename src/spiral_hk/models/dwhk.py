"""DWHK — Dynamical Weight Hegselmann-Krause model grounded in the
Spiral of Silence (Ruan, IEEE UV 2022, doi: 10.1109/UV56588.2022.10185469).

Opinion update (Eq. 2): pairwise convex blends over confidence neighbors

    O_i(t+1) = (1/|N_i|) * sum_{j in N_i} [ (1 - w_ij) O_i(t) + w_ij O_j(t) ]

Weight update (Eq. 6; the camera-ready pseudocode box contains a typo,
"S = 1 - W" — the authoritative closed form is Eq. 6, equivalent to
delta = L W U):

    delta_ij = (2 k_j - d_j) / (n (k_i + 1)) * w_ij,   W <- clip(W + delta, 0, 1)

where k_i = |N_i(t)| counts i's confidence-bounded neighbors and d_i is
i's full degree. Interpretation: k_j measures how locally *majority* j is;
local-majority speakers gain influence, local-minority speakers lose it —
the spiral of silence implemented in weight space. No free parameters are
introduced; `gamma` below is a research extension (gamma = 1 recovers the
paper) used to probe the strength of the climate feedback.

Order per the paper: opinions are updated first (using weights at time t),
then weights are updated (using the same t-step confidence counts k).
"""

from __future__ import annotations

import numpy as np

from spiral_hk.graph import Graph
from spiral_hk.models.hegselmann_krause import confidence_mask


def dwhk_step(
    opinions: np.ndarray,
    weights: np.ndarray,
    g: Graph,
    eps: float,
    gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One synchronous DWHK step.

    Parameters
    ----------
    opinions : (n,) float array, current opinions
    weights  : (m,) float array, per-arc influence weights in [0, 1]
    g        : graph (arc arrays)
    eps      : confidence bound
    gamma    : climate-feedback gain (research extension; 1.0 = paper)

    Returns
    -------
    new_opinions, new_weights, k  where k[i] = |N_i(t)|.
    """
    mask = confidence_mask(opinions, g, eps)
    src, dst = g.src[mask], g.dst[mask]
    w = weights[mask]

    # --- confidence counts k_i (reused by the weight phase) ---
    k = np.bincount(src, minlength=g.n)

    # --- opinion update: O_i' = (1 - mean_j w_ij) O_i + mean_j (w_ij O_j) ---
    sum_w = np.bincount(src, weights=w, minlength=g.n)
    sum_wo = np.bincount(src, weights=w * opinions[dst], minlength=g.n)
    new_opinions = opinions.copy()
    active = k > 0
    new_opinions[active] = (
        opinions[active] * (1.0 - sum_w[active] / k[active])
        + sum_wo[active] / k[active]
    )

    # --- weight update: delta_ij = L_i * U_j * w_ij, clipped to [0, 1] ---
    degree = g.degree
    U = (2.0 * k - degree) / g.n          # local-majority signal of speaker j
    L = 1.0 / (k + 1.0)                   # normalization by listener climate
    delta = L[src] * U[dst] * w
    new_weights = weights.copy()
    new_weights[mask] = np.clip(weights[mask] + gamma * delta, 0.0, 1.0)

    return new_opinions, new_weights, k
