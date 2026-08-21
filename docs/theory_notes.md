# Theory Notes — DWHK Structural Facts

Facts that can be proved on a whiteboard in a few lines and are checked by
`tests/test_models.py`.

## 1. Convex-hull invariance (boundedness)

For any agent i with confidence neighborhood N_i, the DWHK opinion update is

    O_i' = (1/k_i) sum_{j in N_i} [(1 - w_ij) O_i + w_ij O_j]

Expanding, the coefficient of O_i is `(1/k_i) sum_j (1 - w_ij) = 1 - w_bar_i`
and the coefficient of each O_j is `w_ij / k_i`. For w_ij in [0, 1] all
coefficients are non-negative and sum to 1: every update is a **convex
combination**. Hence

    min_j O_j(0) <= O_i(t) <= max_j O_j(0)   for all i, t.

The same argument applies to HK (uniform average) and IHK (convex blend with
inertia). Opinions never leave the initial convex hull.

## 2. Row-stochastic (time-varying DeGroot) form

Define P(t) by

    P_ij = w_ij / k_i        (j in N_i)
    P_ii = 1 - w_bar_i
    P_ij = 0                 otherwise

Then `O(t+1) = P(t) O(t)` and P(t) is row-stochastic. DWHK is therefore a
**time-varying DeGroot system** whose interaction graph and weights are
co-determined by the opinion state. This is the precise hook into consensus
theory:

- If the sequence P(t) is (uniformly) jointly connected and satisfies
  standard bounded-positivity conditions, consensus follows from classical
  results (products of SIA matrices).
- The spiral-of-silence weight update can *break* those conditions
  (minority arcs decay toward 0), which is exactly why the open question is
  interesting: **under what climate feedback does consensus survive, and
  when does the system fragment?** A rigorous answer (convergence theorem +
  phase diagram) is RQ1 of the PhD proposal.

## 3. Per-step weight-change bound

U_jj = (2 k_j - d_j)/n lies in [-d_j/n, d_j/n] ⊂ [-1, 1] and L_ii <= 1, so
|delta_ij| <= w_ij <= 1. Weights are additionally clipped to [0, 1] after
each step (the paper maintains w_ij in [0, 1]).

## 4. Degeneracies (sanity checks)

- All weights w_ij = 1  ⇒  DWHK opinion update == classical HK.
- IHK with inertia λ = 1  ⇒  classical HK.
- gamma = 0 (extension)  ⇒  weights frozen; DWHK becomes HK with fixed
  heterogeneous influence.

## 5. Known erratum in the UV 2022 paper

The Algorithm 1 box prints "S = 1_{n×n} - W", contradicting Eq. 6. Three
independent arguments identify Eq. 6 (delta ∝ w_ij, i.e. delta = L W U) as
the implemented rule:

1. **Algebra**: Eq. 6 expands exactly to (L W U)_ij.
2. **Topology preservation**: results are reported per network (BA/WS/
   ego-Facebook) and the complexity analysis counts edge passes, so W stayed
   sparse throughout; a dense J - W would fully connect the graph after one
   update and make the topology discussion void.
3. **Mechanism**: δ ∝ w_ij implements proportional, gradual silencing
   ("minorities gradually lose voice"); δ ∝ (1 - w_ij) would collapse small
   weights instantly, a hard muting inconsistent with the spiral narrative
   and with initial weights uniform in [0, 1].
