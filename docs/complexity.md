# DWHK Model: Time & Space Complexity — Rigorous Analysis

> Portfolio document (`docs/complexity.md`) + interview whiteboard material.
> This supersedes the wording of Section III in Ruan, IEEE UV 2022.
>
> Notation: G = (V, E), n = |V|, m = |E| (symmetric weighted digraph / WBD);
> O_i^t = opinion of agent i at step t; w_ij = edge weight;
> N_i^t = { j : (i,j) ∈ E, |O_i^t − O_j^t| ≤ ε } (confidence-bounded neighbors);
> k_i = |N_i^t|; d_i = deg(i).

## 1. Model recap

Opinion update (pairwise convex blend over confidence neighbors):

    O_i^{t+1} = (1 / |N_i^t|) · Σ_{j ∈ N_i^t} [ (1 − w_ij)·O_i^t + w_ij·O_j^t ]

Weight update (spiral-of-silence in weight space):

    L = diag( 1/(k_i + 1) ),   U = diag( (2k_j − d_j)/n ),   W ← clip(W + δ, 0, 1)
    δ = L · W · U,  i.e.  δ_ij = (2k_j − d_j) / (n·(k_i + 1)) · w_ij   (Eq. 6)

Known erratum in the UV 2022 camera-ready: the Algorithm 1 box prints
"S = 1_{n×n} − W", which contradicts Eq. 6. The closed form (Eq. 6) is
authoritative and is what the implementation follows. Three independent
arguments: (i) Eq. 6 expands exactly to L·W·U; (ii) results are discussed
per network topology and the complexity section counts edge passes, so W
stayed sparse throughout — a dense J−W would destroy the graph in one step;
(iii) δ ∝ w_ij gives proportional (gradual) silencing consistent with the
spiral-of-silence narrative, whereas δ ∝ (1−w_ij) would collapse small
weights instantly. Weights are clipped to [0, 1] after each update.

## 2. Time complexity per time step

| Phase | Work | Cost (sparse, adjacency lists) |
|---|---|---|
| A1 confidence check | for each edge (i,j): test \|O_i − O_j\| ≤ ε | O(m) |
| A2 opinion accumulation | accumulate (1−w_ij)O_i + w_ij O_j per node | O(m) |
| A3 normalization | divide by \|N_i\|, store k_i for reuse | O(n) |
| B1 degree vector d | precomputed once at t=0 | O(m) once, O(1)/step |
| B2 diagonals L, U | from k and d | O(n) |
| B3 δ and W update | per-edge product and addition, clip to [0,1] | O(m) |
| **Total per step** | | **O(m + n)** |

Variants:
- Dense matrix implementation: O(n²) per step (matrix products; also see §3).
- Fully-mixed (complete-graph) HK variant: sorting opinions once per step gives
  O(n log n), after which confidence neighborhoods are sliding windows, O(n).
- Convergence in T steps ⇒ total O(T·(m + n)).

Note on "2M" in the paper: with WBD each undirected edge is stored as two
directed arcs, so the two passes (opinion phase, weight phase) visit each arc —
consistent with O(m), not an extra asymptotic factor.

## 3. Space complexity

| Item | Cost |
|---|---|
| Opinions O, vectors k, d | O(n) |
| Weight matrix W (edge list) | O(m) |
| δ (computed in place on edges) | O(m) or O(1) in place |
| Sorting scratch (complete-graph variant only) | O(n) |
| **Total** | **O(n + m)** |

**Critical implementation note.** The model uses the Eq. 6 form
(δ_ij ∝ w_ij), so updates touch only existing arcs: sparsity of W is
preserved for all t, no new edges are created, and the O(m + n) per-step
bound holds on any input graph. (Had the camera-ready's literal "S = J − W"
been implemented densely, W would densify within one step and both time and
space would degrade to O(n²) regardless of input sparsity — see §1 erratum.)
Weights are clipped to [0, 1] after each update, since
U_jj ∈ [−d_j/n, d_j/n] can otherwise push weights outside the unit interval.

## 4. Corrections vs. the paper's Section III (defensible phrasing)

1. "Priority queue maintaining agent order by opinion differences" — not
   required by the algorithm as described. The N log N term belongs to TIME
   (one sort per step in the complete-graph variant), not to space.
   Defensible claim: space O(n + m); time O(n log n + m) per step with
   sorted windows, O(m + n) on fixed sparse graphs.
2. "O(M + N log N) space" → O(n + m).
3. "O(2M + N log N) time per step" → O(m + n) per step on sparse graphs
   (the factor 2 reflects two arc passes and is absorbed by O(·)).

## 5. Structural facts (whiteboard gold — provable in 3 lines)

**Fact 1 — Boundedness / convex hull invariance.**
For fixed i, the coefficient of O_i^t is (1/|N_i|)Σ_j (1 − w_ij) = 1 − w̄_i,
and the coefficient of each O_j^t is w_ij/|N_i|. All coefficients are ≥ 0
(for w_ij ∈ [0,1]) and sum to 1. Hence every update is a convex combination:

    min_j O_j^0 ≤ O_i^t ≤ max_j O_j^0   for all i, t.

**Fact 2 — Row-stochastic form.**
Define P(t) with P_ii = 1 − w̄_i and P_ij = w_ij/|N_i^t| for j ∈ N_i^t.
Then O^{t+1} = P(t)·O^t with P(t) row-stochastic — DWHK is a
time-varying DeGroot system. Standard consensus machinery (joint connectivity,
SIA products) is exactly the route to a rigorous convergence theorem
→ PhD RQ1.

**Fact 3 — Weight drift bound.**
U_jj = (2k_j − d_j)/n ∈ [−d_j/n, d_j/n] ⊂ [−1, 1], and L_ii ≤ 1,
so per-step weight changes are bounded: |δ_ij| ≤ 1. Long-run drift without
clipping is O(T) in the worst case — another reason to state the clipping
rule explicitly.

## 6. Interview script (English)

Q: "Walk me through the complexity of your DWHK model."
A: "Per step there are two phases. The opinion phase scans every arc once to
   test the confidence bound and accumulate the pairwise blends — O(m).
   The weight phase reuses the confidence counts k_i, builds two diagonal
   matrices in O(n), and updates weights along existing edges — O(m) again.
   So O(m + n) per step on sparse graphs, O(n²) if you keep the weight
   matrix dense. Opinions stay in the initial convex hull because every
   update is a convex combination, which also lets me write the system as
   O(t+1) = P(t) O(t) with row-stochastic P(t) — the rigorous convergence
   analysis of that time-varying system is what I want to do in my PhD."

Q: "Your paper mentions a priority queue — where is it used?"
A: "That wording in the camera-ready is loose. No heap is needed; the
   n log n term appears only if you sort opinions each step to find
   confidence windows in the fully-mixed case. On a fixed sparse graph the
   tight bound is O(m + n) per step."

Q: "Algorithm 1 in your paper writes S = 1 − W, but Eq. 6 has a factor
    w_ij. Which one did you implement?"
A: "Good catch — that's a typo in the pseudocode box. The implementation
   follows Eq. 6, which expands exactly to L·W·U, so δ_ij ∝ w_ij. Two
   sanity checks confirm it: the experiments rely on the network topology
   surviving all steps — a dense J − W would fully connect the graph after
   one update — and δ ∝ w_ij is the proportional silencing that the
   spiral-of-silence mechanism actually calls for."
