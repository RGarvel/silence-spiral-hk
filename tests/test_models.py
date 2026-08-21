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
from spiral_hk.models import hk_step, ihk_step, dwhk_step
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
