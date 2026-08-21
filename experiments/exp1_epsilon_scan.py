"""exp1 — Classical HK on the complete graph: clusters vs confidence bound.

Reproduces the canonical bounded-confidence phase diagram and marks the
consensus threshold eps_c ~= 0.228 (uniform initial opinions on [0, 1];
Fortunato 2005). This is the baseline any DWHK result is compared against.

Usage:  python experiments/exp1_epsilon_scan.py [--trials 15]
Output: figures/exp1_epsilon_scan.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spiral_hk.graph import complete_graph
from spiral_hk.metrics import n_clusters
from spiral_hk.simulation import run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    g = complete_graph(args.n)
    eps_grid = np.round(np.arange(0.05, 0.51, 0.05), 2)
    rng = np.random.default_rng(42)

    means, stds = [], []
    for eps in eps_grid:
        counts = []
        for _ in range(args.trials):
            opinions0 = rng.uniform(0.0, 1.0, args.n)
            hist = run("hk", g, opinions0, eps=float(eps), steps=args.steps)
            counts.append(n_clusters(hist.final_opinions, eps=float(eps)))
        means.append(np.mean(counts))
        stds.append(np.std(counts))
        print(f"eps={eps:.2f}  clusters={means[-1]:.2f} +/- {stds[-1]:.2f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(eps_grid, means, yerr=stds, marker="o", capsize=3,
                color="tab:blue", label="HK clusters (mean over trials)")
    ax.axvline(0.228, color="tab:red", ls="--", lw=1,
               label=r"consensus threshold $\varepsilon_c \approx 0.228$")
    ax.set_xlabel(r"confidence bound $\varepsilon$")
    ax.set_ylabel("number of opinion clusters")
    ax.set_title(f"HK on complete graph (n={args.n}, uniform opinions)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "figures" / "exp1_epsilon_scan.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
