"""exp5 — does the explicit expression channel produce the spiral-of-silence
signature that DWHK cannot represent?

Signature (Noelle-Neumann): what people *say* converges faster than what they
*think*.  So, with voiced = {i : s_i > threshold}:

    public_std < private_std,   and the gap widens over time.

DWHK cannot express this at all -- it has no s.  The control is eta = 0,
which freezes willingness and must collapse the gap to zero.

Two settings:
  (a) regular graph, controlled 70/30 split across an opinion gap > eps
      (degree uniform -> no structural confound)
  (b) ego-Facebook under the UV 2022 setup

Text output only: matplotlib is not required to run this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx

from spiral_hk.graph import _bidirect, load_snap_edge_file
from spiral_hk.models.sos_extension import private_vs_public_spread, sos_step

THRESHOLD = 0.05
STEPS = 3000


def run(g, O0, W0, eps, eta, steps=STEPS, s0=1.0, checkpoints=(100, 500, 1000, 3000)):
    O = np.asarray(O0, float).copy()
    W = np.asarray(W0, float).copy()
    s = np.full(g.n, float(s0))
    rows = []
    for t in range(1, steps + 1):
        O, W, s, _ = sos_step(O, W, s, g, eps, eta=eta)
        if t in checkpoints:
            m = private_vs_public_spread(O, s, THRESHOLD)
            voiced = s > THRESHOLD
            m["t"] = t
            m["mean_silenced_opinion"] = (
                float(np.mean(O[~voiced])) if (~voiced).any() else float("nan")
            )
            m["mean_voiced_opinion"] = (
                float(np.mean(O[voiced])) if voiced.any() else float("nan")
            )
            m["gap"] = m["private_std"] - m["public_std"]
            rows.append(m)
    return rows, O, W, s


def report(title, rows):
    print(f"\n--- {title} ---")
    print(f"  {'t':>6} {'private_std':>12} {'public_std':>11} {'GAP':>9} "
          f"{'%silenced':>10} {'mean_s':>8} {'O(silent)':>10} {'O(voiced)':>10}")
    for m in rows:
        print(f"  {m['t']:>6} {m['private_std']:>12.4f} {m['public_std']:>11.4f} "
              f"{m['gap']:>+9.4f} {100*m['silenced_fraction']:>9.1f}% "
              f"{m['mean_willingness']:>8.3f} "
              f"{m['mean_silenced_opinion']:>+10.4f} "
              f"{m['mean_voiced_opinion']:>+10.4f}")


def main():
    print("=" * 88)
    print("(a) REGULAR graph, 70/30 split across an opinion gap > eps  "
          "(uniform degree)")
    print("=" * 88)
    n, deg = 600, 10
    G = nx.random_regular_graph(deg, n, seed=3)
    g = _bidirect(n, list(G.edges()), "regular")
    rng = np.random.default_rng(3)
    is_major = np.zeros(n, dtype=bool)
    is_major[: int(0.70 * n)] = True
    rng.shuffle(is_major)
    O0 = np.where(is_major, +0.5, -0.5).astype(float)
    W0 = rng.uniform(0.0, 1.0, g.m)
    for eta in (0.0, 0.05, 0.2):
        rows, O, W, s = run(g, O0, W0, eps=0.3, eta=eta)
        report(f"eta = {eta}  (eta=0 is the DWHK control)", rows)
        silenced = s <= THRESHOLD
        print(f"    final: silenced {int(silenced.sum())}/{n}; "
              f"among them minority = {int((silenced & ~is_major).sum())}, "
              f"majority = {int((silenced & is_major).sum())}")

    print("\n" + "=" * 88)
    print("(b) ego-Facebook, UV 2022 setup (eps = 0.5, O ~ N(0,0.5), W ~ U[0,1])")
    print("=" * 88)
    data = Path(__file__).resolve().parents[1] / "data" / "facebook_combined.txt.gz"
    if not data.exists():
        print(f"  skipped: {data} not found (see README for the download URL)")
        return
    g2 = load_snap_edge_file(str(data))
    rng2 = np.random.default_rng(2022)
    O2 = np.clip(rng2.normal(0.0, 0.5, g2.n), -1.0, 1.0)
    W2 = rng2.uniform(0.0, 1.0, g2.m)
    for eta in (0.0, 0.05, 0.1):
        rows, O, W, s = run(g2, O2, W2, eps=0.5, eta=eta)
        report(f"eta = {eta}  (eta=0 is the DWHK control)", rows)

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print("  * eta = 0  -> no self-censorship: gap must be exactly 0.0")
    print("  * eta > 0  -> public_std < private_std and the gap widens: the")
    print("                signature DWHK cannot represent at all")
    print("  * directionality: is the silenced set the minority side?")


if __name__ == "__main__":
    main()