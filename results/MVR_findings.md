# MVR results vs. Benac et al. (2024) Table 4 — honest read

Run date: 2026-04-13
Scope: minimum viable reproduction (5 seeds for Gridworld, 5 worlds × 2 datasets
for RandomWorld — paper uses 50 and 100 respectively).

## What we ran

- Gridworld 5×5, soft walls [(1,1),(1,2),(2,1),(3,3),(3,2)] with −5 penalty,
  slip 0.2, γ=0.95, standard task.
- RandomWorld 15 states × 5 actions × 5 successors/(s,a), Uniform[0,1] probs,
  state-only rewards, γ=0.95.
- ε=5.0, 40% stochastic-policy-state expert, δ=0.001 Laplace smoothing.
- Coverage sweep {0.2, 0.4, 0.6, 0.8, 1.0} — K=10 (GW) / K=5 (RW) per (s,a).

## Headline numbers (coverage = 1.0)

| metric                  | paper ITL (GW) | paper MLE (GW) | ours ITL (GW) | ours MLE (GW) |
|-------------------------|----------------|----------------|---------------|---------------|
| Best matching (≈)       | 0.56           | 0.31           | **0.488**     | **0.520**     |
| Normalized Value (≈)    | ~0.8–1.0       | ~0.3–0.5       | **+0.097**    | **+0.142**    |

| metric                  | paper ITL (RW) | paper MLE (RW) | ours ITL (RW) | ours MLE (RW) |
|-------------------------|----------------|----------------|---------------|---------------|
| Best matching (≈)       | 0.58           | 0.29           | **0.673**     | **0.700**     |
| Normalized Value (≈)    | ~0.85          | ~0.65          | **0.950**     | **0.965**     |

(Paper numbers above are read off Table 4 from memory — verify against the PDF
before citing.)

## What's off

1. **ITL is not beating MLE in either environment.** On every coverage point,
   MLE matches or slightly beats ITL. The paper's main claim is the opposite.
2. **Gridworld Normalized Value is collapsed (~0.1).** At full coverage the
   recovered policy is only achieving ~10% of V*. With 50% action-matching,
   this implies the wrong half of states include states where a wrong action
   is catastrophic (soft walls / wandering from goal).
3. **RandomWorld absolute best-matching is higher than the paper.** Our MLE at
   coverage=1.0 is ~0.70 vs paper's ~0.29. That's suspicious in the opposite
   direction — either our expert is easier to recover (too many deterministic
   states?) or coverage is implemented differently.
4. **CVXPY prints "solution may be inaccurate"** during the Gridworld ITL
   solve. Worth tightening solver settings / switching to SCS exclusively.

## Most suspicious code right now

`src/itl_solver.py` — the ε-ball is computed from `Q_hat` under the current
`T_hat`, not from *observed expert actions* in the batch data. The paper
defines the ε-ball by which actions the expert was observed to take. Computing
it from `T_hat` (bootstrapped from `T_mle`, which is noisy) is likely why the
QP constraints are mis-specified — they can actively push `T_hat` away from
`T_mle` in the wrong direction, which matches the "ITL worse than MLE" symptom.

Plausible fixes to try, in order:

1. In `solve_itl`, set `valid[s, a] = (N[s, a].sum() > 0)` at visited states
   (treat every expert action at observed states as in-ball) and fall back to
   `Q_hat`-based ε-ball only at unvisited states.
2. Use `T_mle` (not `T_hat`) for `v_lin` throughout Algorithm 1 — the paper's
   linearization trick fixes `v_lin` at the MLE value.
3. Drop Eq-9 (valid-valid) constraints when `|valid_actions[s]| == 1` (they're
   vacuous) and when the state is unvisited (we have no evidence of ambivalence).
4. Tighten OSQP: `eps_abs=1e-8, eps_rel=1e-8, max_iter=50000`.

## What's fine

- Environments look correct against the paper's Figure 5 / RandomWorld spec.
- Expert construction gives 10/25 stochastic states on GW (target 40%) and
  behaves sanely.
- Coverage-based batch generation matches the paper's appendix procedure.
- All six paper metrics are wired up and producing plausible shapes vs coverage.

The scaffolding is right — the solver needs surgery.
