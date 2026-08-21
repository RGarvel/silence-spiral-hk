"""exp2 — Replication of the UV 2022 experiments (Figs. 2-10).

HK vs Inertial HK vs DWHK on BA (scale-free) and WS (small-world) networks,
with the paper's parameters:
  - initial opinions: normal distribution clipped to [-1, 1]
  - initial weights:  uniform in [0, 1]
  - confidence bound eps = 0.5, IHK self-confidence 0.7 (inertia 0.3)
  - 100 time steps

All three models share identical initial conditions per network (same seed).
Output: figures/exp2_<network>.png  (opinion trajectories + convergence)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spiral_hk.graph import Graph, barabasi_albert, watts_strogatz, load_snap_edge_file
from spiral_hk.metrics import convergence_time
from spiral_hk.simulation import run, History

ROOT = Path(__file__).resolve().parents[1]
STEPS = 100
EPS = 0.5


def initial_conditions(g: Graph, seed: int):
    rng = np.random.default_rng(seed)
    opinions = np.clip(rng.normal(0.0, 0.5, g.n), -1.0, 1.0)
    weights = rng.uniform(0.0, 1.0, g.m)
    return opinions, weights


def run_all(g: Graph, seed: int) -> dict[str, History]:
    opinions, weights = initial_conditions(g, seed)
    return {
        "HK": run("hk", g, opinions, EPS, STEPS),
        "IHK": run("ihk", g, opinions, EPS, STEPS, inertia=0.3),
        "DWHK": run("dwhk", g, opinions, EPS, STEPS, weights0=weights),
    }


def plot(g: Graph, results: dict[str, History], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {"HK": "tab:gray", "IHK": "tab:orange", "DWHK": "tab:green"}
    for ax, (name, hist) in zip(axes.flat[:3], results.items()):
        T = len(hist.opinions)
        t_grid = np.arange(T)
        traj = np.stack(hist.opinions, axis=1)  # (n, T)
        ax.plot(t_grid, traj.T, color=colors[name], lw=0.3, alpha=0.25)
        t_conv = convergence_time(hist.max_delta)
        ax.set_title(f"{name}  (converged at t={t_conv})", fontsize=10)
        ax.set_xlabel("time step")
        ax.set_ylabel("opinion")
        ax.set_ylim(-1.05, 1.05)
    ax = axes.flat[3]
    for name, hist in results.items():
        ax.semilogy(hist.max_delta, color=colors[name], label=name)
    ax.axhline(1e-4, color="k", ls=":", lw=1)
    ax.set_xlabel("time step")
    ax.set_ylabel("max opinion change")
    ax.set_title("convergence speed", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f"Opinion evolution on {g.name}  (n={g.n}, m={g.m}, "
                 f"eps={EPS}, UV 2022 setup)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--ego-facebook", type=str, default="",
                    help="path to SNAP facebook_combined.txt(.gz), optional")
    args = ap.parse_args()

    networks = [
        barabasi_albert(1024, 2, seed=1),
        watts_strogatz(1024, 8, 0.1, seed=1),
    ]
    if args.ego_facebook and Path(args.ego_facebook).exists():
        networks.append(load_snap_edge_file(args.ego_facebook))

    outdir = ROOT / "figures"
    outdir.mkdir(exist_ok=True)
    for g in networks:
        print(f"--- {g.name}: n={g.n}, arcs={g.m} ---")
        results = run_all(g, args.seed)
        for name, hist in results.items():
            final = hist.final_opinions
            print(f"  {name:5s}: steps={len(hist.max_delta)}, "
                  f"spread={np.std(final):.3f}, "
                  f"mean={np.mean(final):+.3f}")
        plot(g, results, outdir / f"exp2_{g.name.split('(')[0]}.png")


if __name__ == "__main__":
    main()
