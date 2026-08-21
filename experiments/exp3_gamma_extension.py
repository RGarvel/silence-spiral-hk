"""exp3 — Research extension: strength of the climate feedback (gamma).

DWHK recovers the paper when gamma = 1. Sweeping gamma probes how strongly
the spiral-of-silence feedback shapes outcomes — the "bifurcation-style"
figure proposed for the PhD agenda (RQ1):

  - gamma = 0:  static weights (weights frozen at initialization)
  - gamma = 1:  the UV 2022 model
  - gamma > 1:  amplified climate feedback

Measured outcomes: dominant-cluster share, silenced-edge fraction,
convergence time. Output: figures/exp3_gamma_sweep.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spiral_hk.graph import barabasi_albert
from spiral_hk.metrics import (convergence_time, largest_cluster_share,
                               silenced_edge_fraction)
from spiral_hk.simulation import run

ROOT = Path(__file__).resolve().parents[1]
EPS = 0.5
STEPS = 200
GAMMAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args()

    g = barabasi_albert(args.n, 2, seed=1)
    dominance, silenced, conv = [], [], []

    for gamma in GAMMAS:
        d, s, c = [], [], []
        for trial in range(args.trials):
            rng = np.random.default_rng(100 + trial)
            opinions0 = np.clip(rng.normal(0.0, 0.5, g.n), -1.0, 1.0)
            weights0 = rng.uniform(0.0, 1.0, g.m)
            hist = run("dwhk", g, opinions0, EPS, STEPS,
                       weights0=weights0, gamma=gamma)
            d.append(largest_cluster_share(hist.final_opinions, EPS))
            s.append(silenced_edge_fraction(hist.weights[-1]))
            c.append(convergence_time(hist.max_delta))
        dominance.append((np.mean(d), np.std(d)))
        silenced.append((np.mean(s), np.std(s)))
        conv.append((np.mean(c), np.std(c)))
        print(f"gamma={gamma:4.2f}  dominance={dominance[-1][0]:.3f} "
              f"silenced={silenced[-1][0]:.3f}  conv_t={conv[-1][0]:.1f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, series, title in zip(
        axes,
        (dominance, silenced, conv),
        ("dominant cluster share", "silenced edge fraction (w<0.1)",
         "convergence time (steps)"),
    ):
        means = [m for m, _ in series]
        stds = [s for _, s in series]
        ax.errorbar(GAMMAS, means, yerr=stds, marker="o", capsize=3,
                    color="tab:purple")
        ax.axvline(1.0, color="tab:red", ls="--", lw=1)
        ax.set_xlabel(r"climate-feedback gain $\gamma$ (paper: $\gamma=1$)")
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"DWHK on {g.name} (n={g.n}, eps={EPS}) — "
                 "extension beyond Ruan 2022", fontsize=10)
    fig.tight_layout()
    out = ROOT / "figures" / "exp3_gamma_sweep.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
