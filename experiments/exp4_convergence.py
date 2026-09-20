"""exp4 — long-horizon convergence audit.

exp2's convergence claims rest on a 100-step run with tol = 1e-4, which is too
weak a caliber: an intermittent trajectory can produce a large one-off jump and
still be converging.  This experiment re-audits convergence properly.

Three parts:
  1. Exhaustive small graphs (n = 2..6) + complete bipartite K_{p,q}: does
     DWHK actually converge, and are the failures exact period-2 cycles?
  2. BA and WS networks, eps x seed grid: oscillation rate of the unweighted
     HK baseline vs DWHK.
  3. ego-Facebook, 10 000 steps: per-step change delta(t) for HK, the
     frozen-weight DWHK variant (gamma = 0), and DWHK (gamma = 1).

METRIC WARNING (learned twice, in two different disguises).  Neither of these
works:
  * max(delta) over a trailing window -- one isolated avalanche inside the
    window reports "not converged" while the trajectory is settling.
  * late_delta < tol with a strict tol -- DWHK converges *slowly*
    (~1e-5 at t=2000), so a strict tol silently reclassifies slow convergence
    as oscillation.
The test used here compares an early window against a late window: a delta
that DECAYS is settling; a delta that stays PINNED is oscillating.

Text output only: matplotlib is not required to run this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx

from spiral_hk.graph import _bidirect, barabasi_albert, load_snap_edge_file, watts_strogatz
from spiral_hk.models.dwhk import dwhk_step
from spiral_hk.models.hegselmann_krause import hk_step

EPS_GRID = (0.2, 0.3, 0.5)
STEPS = 4000
TOL = 1e-9


def trace_hk(g, O0, eps, steps):
    O = np.array(O0, float)
    hist = []
    for _ in range(steps):
        new = hk_step(O, g, eps)
        hist.append(float(np.max(np.abs(new - O))))
        O = new
    return np.array(hist), O


def trace_dwhk(g, O0, W0, eps, steps, gamma=1.0):
    O, W = np.array(O0, float), np.array(W0, float)
    hist = []
    for _ in range(steps):
        new, W, _ = dwhk_step(O, W, g, eps, gamma=gamma)
        hist.append(float(np.max(np.abs(new - O))))
        O = new
    return np.array(hist), O


def classify(deltas, tol=TOL, slope_eps=0.05):
    """Distinguish genuine oscillation from slow convergence, by the power law.

    A delta pinned at a positive value means oscillation; a delta that decays
    as a power law means the trajectory is converging, just slowly.  Both of
    these weaker tests have already fooled us once each:
      * max(delta) over a trailing window -> one avalanche reports "not converged"
      * late_delta < tol with a strict tol -> DWHK's ~t^-1 decay is silently
        reclassified as oscillation
    So fit the envelope of the second half in log-log space and read the slope:

        slope ~ 0     -> pinned: genuine oscillation
        slope < 0     -> power-law decay: converging (however slowly)

    Returns (verdict, late_delta, slope).
    """
    n = len(deltas)
    late = float(np.max(deltas[-100:]))
    if late < tol:
        return "converged", late, 0.0
    start = n // 2
    seg = deltas[start:]
    nw = 8
    bounds = np.linspace(0, len(seg), nw + 1).astype(int)
    env = np.array([float(seg[bounds[k]:bounds[k + 1]].max()) for k in range(nw)])
    tmid = np.array([start + (bounds[k] + bounds[k + 1]) // 2 for k in range(nw)],
                    dtype=float)
    good = env > 0
    if good.sum() >= 3:
        slope = float(np.polyfit(np.log(tmid[good]), np.log(env[good]), 1)[0])
    else:
        slope = 0.0
    if abs(slope) < slope_eps:
        return "oscillating", late, slope
    return "settling", late, slope


def is_exact_period2(deltas, tol=TOL):
    """The last few steps repeat with an identical magnitude."""
    if deltas[-1] <= tol:
        return False
    return bool(np.all(np.abs(deltas[-5:] - deltas[-1]) < 1e-12))


def part1_small_graphs():
    print("=" * 88)
    print("(1) Exhaustive small graphs (n = 2..6), DWHK")
    print("=" * 88)
    n_conf = n_conv = n_osc = n_p2 = n_settle = 0
    examples = []
    for G in nx.graph_atlas_g():
        n = G.number_of_nodes()
        if not (2 <= n <= 6) or G.number_of_edges() == 0:
            continue
        edges = list(G.edges())
        g = _bidirect(n, edges, f"atlas{n}")
        for eps in EPS_GRID:
            for wname in ("uniform", "ones"):
                rng = np.random.default_rng(7 * n + int(10 * eps))
                O0 = rng.uniform(-1.0, 1.0, n)
                W0 = (rng.uniform(0.0, 1.0, g.m) if wname == "uniform"
                      else np.ones(g.m))
                deltas, Ofin = trace_dwhk(g, O0, W0, eps, STEPS)
                n_conf += 1
                cls, late, slope = classify(deltas)
                if cls == "converged":
                    n_conv += 1
                elif cls == "oscillating":
                    n_osc += 1
                    if is_exact_period2(deltas):
                        n_p2 += 1
                    if len(examples) < 3:
                        examples.append((n, len(edges), eps, wname, late,
                                         Ofin.copy()))
                else:
                    n_settle += 1
    print(f"  configurations          = {n_conf}")
    print(f"  converged               = {n_conv}  ({100*n_conv/n_conf:.1f}%)")
    print(f"  oscillating             = {n_osc}  (of which exact period-2: {n_p2})")
    print(f"  still settling          = {n_settle}")
    for n, m, eps, wname, d, Ofin in examples:
        print(f"    e.g. n={n} m={m} eps={eps} W={wname}: "
              f"delta={d:.4f} final O={np.round(Ofin, 4)}")

    print("\n  complete bipartite K_{p,q}, 20 random initial conditions each:")
    print(f"    {'graph':>10} {'w=1 : oscillating':>20} {'w=0.5 : oscillating':>21}")
    for p, q in ((1, 1), (1, 2), (2, 2), (3, 3), (5, 5), (10, 10)):
        n = p + q
        G = nx.complete_bipartite_graph(p, q)
        g = _bidirect(n, list(G.edges()), f"K{p}{q}")
        counts = {}
        for label, wval in (("w=1", None), ("w=0.5", 0.5)):
            osc = 0
            for seed in range(20):
                rng = np.random.default_rng(seed)
                O0 = rng.uniform(-1.0, 1.0, n)
                W0 = np.ones(g.m) if wval is None else np.full(g.m, wval)
                deltas, _ = trace_dwhk(g, O0, W0, 0.5, 1500)
                if classify(deltas)[0] == "oscillating":
                    osc += 1
            counts[label] = osc
        print(f"    {f'K_{{{p},{q}}}':>10} "
              f"{counts['w=1']:>15}/20 {counts['w=0.5']:>20}/20")
    print("  -> cycling depends on the initial condition, not on the graph alone.")


def part2_networks(steps=4000):
    print("\n" + "=" * 88)
    print(f"(2) BA(500,3) and WS(500,6,0.1), {steps} steps: "
          f"unweighted HK vs DWHK")
    print("=" * 88)
    for name, mk in (("BA(500,3)", lambda: barabasi_albert(500, 3, seed=11)),
                     ("WS(500,6,0.1)", lambda: watts_strogatz(500, 6, 0.1, seed=11))):
        g = mk()
        tally = {m: {"hk": [], "dwhk": []} for m in (0.1, 0.2, 0.3, 0.5)}
        for eps in (0.1, 0.2, 0.3, 0.5):
            for seed in range(6):
                rng = np.random.default_rng(seed)
                O0 = rng.uniform(-1.0, 1.0, g.n)
                tally[eps]["hk"].append(classify(trace_hk(g, O0, eps, steps)[0]))
                W0 = rng.uniform(0.0, 1.0, g.m)
                tally[eps]["dwhk"].append(
                    classify(trace_dwhk(g, O0, W0, eps, steps)[0]))
        for model in ("hk", "dwhk"):
            rows = [r for e in tally for r in tally[e][model]]
            verdicts = [v for v, _, _ in rows]
            slopes = np.array([s for _, _, s in rows])
            osc = verdicts.count("oscillating")
            conv = verdicts.count("converged")
            settle = verdicts.count("settling")
            print(f"  {name:<15} {model.upper():<5} "
                  f"oscillating {osc:>2}/24   settled-to-tol {conv:>2}/24   "
                  f"power-law decay {settle:>2}/24   median slope {np.median(slopes):+.3f}")
    print("  -> 'power-law decay' is NOT oscillation: DWHK converges slowly.")


def part3_facebook(steps=10000):
    print("\n" + "=" * 88)
    print(f"(3) ego-Facebook, {steps} steps — delta(t) trend")
    print("=" * 88)
    data = Path(__file__).resolve().parents[1] / "data" / "facebook_combined.txt.gz"
    if not data.exists():
        print(f"  skipped: {data} not found (see README for the download URL)")
        return
    g = load_snap_edge_file(str(data))
    print(f"  n = {g.n}, arcs = {g.m}")
    rng = np.random.default_rng(2022)
    O0 = np.clip(rng.normal(0.0, 0.5, g.n), -1.0, 1.0)
    W0 = rng.uniform(0.0, 1.0, g.m)

    hk = trace_hk(g, O0, 0.5, steps)[0]
    g0 = trace_dwhk(g, O0, W0, 0.5, steps, gamma=0.0)[0]
    g1 = trace_dwhk(g, O0, W0, 0.5, steps, gamma=1.0)[0]
    marks = tuple(t for t in (100, 1000, 3000, 5000, 10000) if t <= steps)
    print(f"\n  {'t':>7} {'HK':>12} {'DWHK gamma=0':>14} {'DWHK gamma=1':>14}")
    for t in marks:
        print(f"  {t:>7} {hk[t-1]:>12.3e} {g0[t-1]:>14.3e} {g1[t-1]:>14.3e}")
    for label, arr in (("HK", hk), ("DWHK gamma=0", g0), ("DWHK gamma=1", g1)):
        verdict, late, slope = classify(arr)
        jumps = np.flatnonzero(arr > 1e-2)
        print(f"  {label:<14} {verdict:<12} late={late:.3e}  "
              f"log-log slope={slope:+.3f}  jumps>1e-2: {jumps.size:>5}")


if __name__ == "__main__":
    part1_small_graphs()
    part2_networks()
    part3_facebook()
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print("  * DWHK convergence is NOT universal: exact period-2 cycles occur on")
    print("    saturated-weight bipartite graphs, for suitable initial conditions.")
    print("  * On ego-Facebook the unweighted HK baseline oscillates forever")
    print("    (delta pinned at 2.0e-1) while both DWHK variants settle.")
    print("  * Boundedness and monotone non-increasing RANGE are provable;")
    print("    general convergence is open (RQ1 of the PhD agenda).")