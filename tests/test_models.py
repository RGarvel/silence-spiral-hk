"""Unit tests: structural facts of the models (whiteboard material).

Covers:
- convex-hull invariance of every model (each update is a convex blend)
- row-stochastic structure of the DWHK opinion update
- degeneracies: IHK(lam=1) == HK; DWHK(w=1) == HK
- weight clipping keeps W in [0, 1]
"""

import numpy as np
import pytest

from spiral_hk.graph import Graph, complete_graph, barabasi_albert
from spiral_hk.models import (
    dwhk_step,
    hk_step,
    ihk_step,
    perceived_climate,
    sos_step,
)
from spiral_hk.simulation import run


@pytest.fixture
def graph() -> Graph:
    return barabasi_albert(60, 2, seed=7)


@pytest.fixture
def opinions(graph: Graph) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.uniform(-1.0, 1.0, graph.n)


def test_convex_hull_invariance_dwhk(graph, opinions):
    lo, hi = opinions.min(), opinions.max()
    weights = np.random.default_rng(1).uniform(0.0, 1.0, graph.m)
    hist = run("dwhk", graph, opinions, eps=0.5, steps=30, weights0=weights)
    for o in hist.opinions:
        assert o.min() >= lo - 1e-12 and o.max() <= hi + 1e-12


def test_convex_hull_invariance_hk_ihk(graph, opinions):
    lo, hi = opinions.min(), opinions.max()
    for model in ("hk", "ihk"):
        hist = run(model, graph, opinions, eps=0.5, steps=30)
        for o in hist.opinions:
            assert o.min() >= lo - 1e-12 and o.max() <= hi + 1e-12


def test_ihk_lambda1_equals_hk(graph, opinions):
    assert np.allclose(ihk_step(opinions, graph, 0.5, inertia=1.0),
                       hk_step(opinions, graph, 0.5))


def test_dwhk_unit_weights_equal_hk(graph, opinions):
    """With w_ij = 1 for all arcs, DWHK opinions coincide with HK."""
    weights = np.ones(graph.m)
    new_o, _, _ = dwhk_step(opinions, weights, graph, eps=0.5)
    assert np.allclose(new_o, hk_step(opinions, graph, 0.5))


def test_weights_stay_in_unit_interval(graph, opinions):
    rng = np.random.default_rng(5)
    weights = rng.uniform(0.0, 1.0, graph.m)
    for gamma in (1.0, 4.0):  # strong feedback must still be clipped
        hist = run("dwhk", graph, opinions, eps=0.5, steps=40,
                   weights0=weights, gamma=gamma)
        for w in hist.weights:
            assert w.min() >= 0.0 and w.max() <= 1.0


def test_row_stochastic_structure(graph, opinions):
    """O(t+1) = P(t) O(t) with P row-stochastic (-> time-varying DeGroot)."""
    n = graph.n
    eps = 0.5
    weights = np.random.default_rng(2).uniform(0.0, 1.0, graph.m)
    mask = np.abs(opinions[graph.src] - opinions[graph.dst]) <= eps
    P = np.zeros((n, n))
    for (i, j), w in zip(zip(graph.src[mask], graph.dst[mask]), weights[mask]):
        P[i, j] += w
    k = np.bincount(graph.src[mask], minlength=n)
    for i in range(n):
        if k[i] > 0:
            P[i] /= k[i]
            P[i, i] += 1.0 - P[i].sum() + 0.0  # (1-w_bar) on the diagonal
    # rows with k=0 are identity rows
    for i in range(n):
        if k[i] == 0:
            P[i, i] = 1.0
    assert np.allclose(P.sum(axis=1), 1.0)
    new_o, _, _ = dwhk_step(opinions, weights, graph, eps)
    assert np.allclose(P @ opinions, new_o)


# ---------------------------------------------------------------------------
# DWHK-SOS extension
# ---------------------------------------------------------------------------
def test_sos_eta0_s1_reproduces_dwhk_exactly(graph, opinions):
    """Strict generalisation: with eta = 0 and s = 1 the SOS step must
    reproduce DWHK exactly.

    The opinion update is algebraically identical but re-associated
    (DWHK: O*(1-w_bar) + mean(w O_j); SOS: O + mean(w (O_j - O))), so the
    residual is floating-point noise at machine epsilon -- measured at
    ~1e-16 and non-accumulating over thousands of steps.  Weights are
    bit-identical because that code path is shared.
    """
    weights = np.random.default_rng(11).uniform(0.0, 1.0, graph.m)
    s = np.ones(graph.n)
    o_d, w_d, _ = dwhk_step(opinions, weights, graph, eps=0.5, gamma=1.0)
    o_s, w_s, s_s, _ = sos_step(opinions, weights, s, graph, eps=0.5, eta=0.0)
    assert np.allclose(o_s, o_d, rtol=0.0, atol=1e-14), "opinions diverge from DWHK"
    assert np.array_equal(w_s, w_d), "weights must be bit-identical"
    assert np.array_equal(s_s, s), "willingness must be frozen at eta = 0"


def test_sos_convex_hull_invariance(graph, opinions):
    """The private-opinion update stays inside the initial convex hull."""
    lo, hi = opinions.min(), opinions.max()
    rng = np.random.default_rng(13)
    weights = rng.uniform(0.0, 1.0, graph.m)
    s = rng.uniform(0.0, 1.0, graph.n)
    o = opinions.copy()
    for _ in range(60):
        o, weights, s, _ = sos_step(o, weights, s, graph, eps=0.5, eta=0.1)
        assert o.min() >= lo - 1e-12 and o.max() <= hi + 1e-12


def test_sos_willingness_stays_in_unit_interval(graph, opinions):
    rng = np.random.default_rng(17)
    weights = rng.uniform(0.0, 1.0, graph.m)
    s = rng.uniform(0.0, 1.0, graph.n)
    o = opinions.copy()
    for _ in range(80):
        o, weights, s, _ = sos_step(o, weights, s, graph, eps=0.5, eta=0.5)
        assert s.min() >= 0.0 and s.max() <= 1.0


def test_perceived_climate_is_neighbourhood_mean(graph, opinions):
    """With s = 1 the perceived climate is the plain mean of the FULL
    graph neighbourhood (not the confidence-limited one)."""
    weights = np.ones(graph.m)
    s = np.ones(graph.n)
    ehat = perceived_climate(opinions, s, weights, graph)
    for i in range(graph.n):
        nb = graph.dst[graph.src == i]
        if nb.size:
            assert np.isclose(ehat[i], opinions[nb].mean())
        else:
            assert ehat[i] == opinions[i]
