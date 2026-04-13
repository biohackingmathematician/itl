# MVR results vs. Benac et al. (2024) Table 4

Run date: 2026-04-13
Scope: minimum viable reproduction — 5 seeds per (coverage, env) on Gridworld
and 10 runs per coverage on RandomWorld (paper uses 50 and 100 respectively).

## Setup

- Gridworld 5×5, soft walls [(1,1),(1,2),(2,1),(3,3),(3,2)] with −5 penalty,
  slip 0.2, γ=0.95, standard task, start bottom-left, goal top-right.
- RandomWorld 15 states × 5 actions × 5 successors/(s,a), Uniform[0,1] probs
  normalized, state-only rewards, γ=0.95, uniform initial distribution.
- ε=5.0, 40% stochastic-policy-state expert, δ=0.001 Laplace smoothing.
- Coverage ∈ {0.2, 0.4, 0.6, 0.8, 1.0}. K=10 (GW) / K=5 (RW) per observed (s,a).

## Headline results (coverage = 1.0)

### Gridworld

| metric               | paper ITL     | paper MLE     | ours ITL      | ours MLE      |
|----------------------|---------------|---------------|---------------|---------------|
| Best matching        | 0.56 ± 0.11   | 0.31 ± 0.06   | **0.816 ± 0.083** | 0.520 ± 0.063 |
| ε-matching           | —             | —             | **1.000 ± 0.000** | 0.712 ± 0.018 |
| Normalized Value     | ≈1.0 (Fig 2)  | ≈0.2–0.4      | **0.998 ± 0.001** | 0.142 ± 0.002 |
| Constraint violations | ≈0            | >0            | **0.00**          | 7.20          |

### RandomWorld

| metric               | paper ITL     | paper MLE     | ours ITL      | ours MLE      |
|----------------------|---------------|---------------|---------------|---------------|
| Best matching        | 0.58 ± 0.15   | 0.29 ± 0.11   | **0.773 ± 0.078** | 0.700 ± 0.131 |
| ε-matching           | —             | —             | **1.000**         | 0.987         |
| Normalized Value     | ≈0.9 (Fig 2)  | ≈0.7          | **0.984 ± 0.010** | 0.965 ± 0.023 |

## What matches the paper

1. **ITL ≥ MLE on every metric at every coverage point** across both
   environments. This is the paper's central empirical claim.
2. **ε-matching goes to 1.0 for ITL at coverage = 1.0** in both environments
   — Theorem 1 predicts ITL recovers π\* exactly when the ε-ball property
   holds at every state.
3. **Constraint violations drop to zero for ITL** at high coverage in
   Gridworld (MLE still averages 7.2 violations at coverage = 1.0). Matches
   the paper's statement that "ITL and BITL do not violate any constraints".
4. **Normalized Value curve monotonically increases with coverage for ITL**
   and plateaus near 1.0 — Figure 2 top-left in the paper.
5. **MLE's Gridworld normalized value is collapsed** (~0.14 at coverage 1.0).
   This is the paper's Fig 2 story — MLE picks plausible-but-wrong greedy
   actions that walk into soft walls repeatedly, so even with correct-ish
   argmax at half the states, V\_MLE/V\* is tiny.

## What doesn't match

1. **Absolute best-matching numbers are ~1.5× paper's**, for both ITL and
   MLE. Most likely: my coverage semantics ("pick ⌈coverage·|S|⌉ states
   uniformly, observe K samples per observed (s,a)") gives more uniform
   coverage than the paper's appendix procedure, which seems to stratify by
   expert visitation. Check Appendix Section "Data generation" before
   submitting results to anyone.
2. **Gridworld seeds 0.4 and 0.6 have wide std on normalized value**
   (±0.43 and ±0.35 respectively) — a few seeds land the expert in a basin
   where even ITL can't fully recover with partial coverage. With 5 seeds
   this noise dominates; 50-seed averaging should smooth it.

## What changed in the solver

See `src/itl_solver.py` commit for the three fixes to Algorithm 1 / Eq 10:

1. **ε-ball now comes from observed expert actions** (per the paper's
   Definition 1 + "Constraints given batch data D only" section). Previously
   derived from Q\_hat under T\_hat, which bootstraps noise.
2. **Constraints are added only at observed states** (per Eq 10's
   `∀(s,a) ∈ D`). Previously applied at every state, which pollutes the QP
   at unvisited states.
3. **Initial policy π(0) for s ∉ D is uniform** (Algorithm 1 line 3).
   Previously greedy-under-T\_MLE.
4. **Linearization always uses T\_MLE** in the matrix inverse (Eq 10's
   "linearization trick"). The old code was passing T\_MLE too — kept this
   for clarity.
5. **Termination when ε-ball property holds at every observed state**
   (Algorithm 1 line 7). Previously only a numerical fixed-point check.
6. **Solver fallback chain OSQP → SCS → CLARABEL** with tighter tolerances,
   eliminating the "solution may be inaccurate" warnings.

## Next steps (not done in this pass)

- Scale seeds up to 50 (paper) — will need background job or external runner.
- Add transfer-task evaluation (swap reward sign / relocate walls, eval same
  T\_hat).
- Verify BITL HMC-with-reflection path (`run_bitl.py`) runs cleanly; it
  wasn't exercised in this pass.
- Add MCE baseline from (Herman et al. 2016) to reproduce Table 4 in full.
- Sanity-check expert generation against the paper's appendix: specifically,
  the way the paper picks which states become stochastic-policy states (I
  pick the N states with smallest Q-gap; paper may pick differently).
