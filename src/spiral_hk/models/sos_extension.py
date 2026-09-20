"""DWHK-SOS — the DWHK model with the spiral of silence's *expression*
channel made explicit.

Motivation
----------
DWHK encodes the spiral of silence in **influence weights**: listeners
down-weight local-minority speakers.  Noelle-Neumann's original mechanism is
different -- the minority **self-censors**: it stops *speaking*.  The two have
similar effects on the opinion distribution but different observable
implications, and the empirical literature tests the *second* channel
(willingness to speak out; see Glynn, Hayes & Shanahan 1997, Public Opinion
Quarterly, doi:10.1086/297808).

This module adds that channel.  Each agent carries

    O_i : private opinion
    s_i : willingness to speak out, in [0, 1]

and the two channels are separated cleanly:

    SEES   -- perception uses the FULL graph neighbourhood, weighted by how
              loudly each neighbour speaks (w_ij * s_j).  This is the
              "climate of opinion" an agent perceives.
    HEARS  -- persuasion uses the confidence-limited neighbourhood N_i
              (|O_i - O_j| <= eps), as in DWHK, and a speaker is audible
              only in proportion to s_j.

Strict generalisation
---------------------
With ``eta = 0`` and ``s0 = 1`` this step reproduces ``dwhk_step`` **exactly**
(same opinions and same weights), because then

    O_i' = O_i + (1/k_i) sum_{j in N_i} w_ij (O_j - O_i)

which is DWHK's Eq. 2 rewritten.  ``tests/test_models.py`` asserts this.

Why perception is NOT the confidence neighbourhood
--------------------------------------------------
If the perceived climate were computed only over N_i, then |O_i - Ehat_i|
<= eps would hold automatically and the agent could never feel isolated --
willingness would never decrease and the mechanism would be dead on arrival.
Perception therefore uses the full neighbourhood, which is also what the
theory says: the climate of opinion is the *general* perceived majority, not
the set of people you already agree with.

Cost
----
DWHK has no free parameters.  Making expression explicit is not free: the
willingness rule introduces one new parameter, ``eta`` (the adaptation rate).
``s0 = 1`` adds nothing.  That trade -- one parameter for the channel that the
empirical literature actually measures -- is deliberate.
"""

from __future__ import annotations

import numpy as np

from spiral_hk.graph import Graph
from spiral_hk.models.hegselmann_krause import confidence_mask


def perceived_climate(
    opinions: np.ndarray, willingness: np.ndarray, weights: np.ndarray, g: Graph
) -> np.ndarray:
    """Ehat_i = audibility-weighted mean opinion over i's FULL neighbourhood.

    Audibility of speaker j is w_ij * s_j.  Agents who hear nobody keep their
    own opinion as their perceived climate (denominator 0 -> Ehat_i = O_i).
    """
    aud = weights * willingness[g.dst]                 # (m,) w_ij * s_j
    num = np.bincount(g.src, weights=aud * opinions[g.dst], minlength=g.n)
    den = np.bincount(g.src, weights=aud, minlength=g.n)
    ehat = opinions.copy()
    heard = den > 0.0
    ehat[heard] = num[heard] / den[heard]
    return ehat


def sos_step(
    opinions: np.ndarray,
    weights: np.ndarray,
    willingness: np.ndarray,
    g: Graph,
    eps: float,
    eta: float = 0.05,
    gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One synchronous DWHK-SOS step.

    Parameters
    ----------
    opinions    : (n,) private opinions
    weights     : (m,) per-arc influence weights in [0, 1]
    willingness : (n,) willingness to speak out, in [0, 1]
    eps         : confidence bound
    eta         : willingness adaptation rate (eta = 0 freezes s -> DWHK)
    gamma       : climate-feedback gain (1.0 = the UV 2022 paper)

    Returns
    -------
    new_opinions, new_weights, new_willingness, k   with k[i] = |N_i(t)|.
    """
    # --- confidence-limited persuasion neighbourhood (as in DWHK) ---
    mask = confidence_mask(opinions, g, eps)
    src, dst = g.src[mask], g.dst[mask]
    w = weights[mask]
    k = np.bincount(src, minlength=g.n)

    # --- willingness: self-censorship driven by the perceived climate ---
    ehat = perceived_climate(opinions, willingness, weights, g)
    if eta != 0.0:
        # d_i = 0 -> agent sits exactly in the perceived climate -> speaks up
        # d_i = eps or more -> agent feels isolated -> falls silent
        drive = 1.0 - np.abs(opinions - ehat) / eps
        new_willingness = np.clip(willingness + eta * drive, 0.0, 1.0)
    else:
        new_willingness = willingness.copy()

    # --- private opinion: DWHK, with each speaker audible in proportion to s_j ---
    aud = w * willingness[dst]                          # w_ij * s_j on N_i
    pull = np.bincount(src, weights=aud * (opinions[dst] - opinions[src]), minlength=g.n)
    new_opinions = opinions.copy()
    active = k > 0
    new_opinions[active] = opinions[active] + pull[active] / k[active]

    # --- influence weights: unchanged DWHK Eq. 6 ---
    U = (2.0 * k - g.degree) / g.n
    L = 1.0 / (k + 1.0)
    delta = L[src] * U[dst] * w
    new_weights = weights.copy()
    new_weights[mask] = np.clip(weights[mask] + gamma * delta, 0.0, 1.0)

    return new_opinions, new_weights, new_willingness, k


# ---------------------------------------------------------------------------
# Observable quantities that only exist once expression is explicit
# ---------------------------------------------------------------------------
def voiced_mask(willingness: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Agents who actually speak: s_i above `threshold`."""
    return willingness > threshold


def private_vs_public_spread(
    opinions: np.ndarray, willingness: np.ndarray, threshold: float = 0.05
) -> dict:
    """The signature the spiral of silence predicts.

    private_std : spread of what people *think* (everyone)
    public_std  : spread of what is actually *said* (voiced agents only)
    The theory predicts public_std < private_std, and a widening gap.
    """
    voiced = voiced_mask(willingness, threshold)
    return {
        "private_std": float(np.std(opinions)),
        "public_std": float(np.std(opinions[voiced])) if voiced.any() else 0.0,
        "silenced_fraction": float(np.mean(~voiced)),
        "mean_willingness": float(np.mean(willingness)),
    }