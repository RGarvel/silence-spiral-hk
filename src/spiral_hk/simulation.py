"""Simulation driver shared by all models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spiral_hk.graph import Graph
from spiral_hk.models.hegselmann_krause import hk_step
from spiral_hk.models.inertial_hk import ihk_step
from spiral_hk.models.dwhk import dwhk_step


@dataclass
class History:
    """Recorded trajectory of a simulation run."""

    opinions: list[np.ndarray] = field(default_factory=list)
    weights: list[np.ndarray] = field(default_factory=list)
    k: list[np.ndarray] = field(default_factory=list)
    max_delta: list[float] = field(default_factory=list)

    @property
    def final_opinions(self) -> np.ndarray:
        return self.opinions[-1]


def run(
    model: str,
    g: Graph,
    opinions0: np.ndarray,
    eps: float,
    steps: int,
    weights0: np.ndarray | None = None,
    inertia: float = 0.3,
    gamma: float = 1.0,
    tol: float = 1e-7,
) -> History:
    """Run `model` in {"hk", "ihk", "dwhk"} for up to `steps` steps.

    Stops early once the maximal opinion change falls below `tol`.
    For "dwhk", `weights0` must be provided (per-arc, in [0, 1]).
    """
    opinions = np.asarray(opinions0, dtype=float).copy()
    weights = None if weights0 is None else np.asarray(weights0, dtype=float).copy()
    if model == "dwhk" and weights is None:
        raise ValueError("dwhk requires weights0")

    hist = History()
    hist.opinions.append(opinions.copy())
    if weights is not None:
        hist.weights.append(weights.copy())

    for _ in range(steps):
        if model == "hk":
            new_opinions = hk_step(opinions, g, eps)
        elif model == "ihk":
            new_opinions = ihk_step(opinions, g, eps, inertia=inertia)
        elif model == "dwhk":
            new_opinions, weights, k = dwhk_step(opinions, weights, g, eps, gamma=gamma)
            hist.k.append(k)
            hist.weights.append(weights.copy())
        else:
            raise ValueError(f"unknown model: {model}")

        delta = float(np.max(np.abs(new_opinions - opinions))) if g.n else 0.0
        hist.max_delta.append(delta)
        opinions = new_opinions
        hist.opinions.append(opinions.copy())
        if delta < tol:
            break

    return hist
