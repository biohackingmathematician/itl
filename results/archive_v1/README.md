# archive_v1: First-pass results (DO NOT USE FOR COMPARISON TO PAPER)

These are the results produced by the initial scaffold (commit `ab4c123`, Apr 13 2026). Kept for before/after comparison only.

**Why these don't match the paper:**

1. **Wrong metric.** Reports transition-matrix MSE. The paper (Benac et al. 2024, Table 4) reports policy-recovery metrics: Best matching, ε-matching, Normalized Value, Bayesian Regret. ITL's advantage is far larger on the paper's metrics than on MSE.

2. **Wrong γ.** Used 0.9; paper uses 0.95.

3. **No Laplace smoothing.** Paper uses δ = 0.001 in the MLE (Eq 5).

4. **Gridworld reward function incomplete.** Missing the soft-wall tiles (-5 penalty) that are central to both the standard and transfer tasks. Start/goal also wrong (paper: start bottom-left, goal top-right).

5. **RandomWorld wrong.** Used Dirichlet transitions and N(0,1) state-action rewards. Paper: 5 successors with probabilities drawn from Uniform[0,1] and normalized; rewards Uniform[16-s-1, 16-s] depending only on state.

6. **No coverage sweep.** Paper's Figure 2 is Normalized Value vs. coverage %. Current code sweeps ε instead. Data generation also didn't subsample states by coverage.

7. **Wrong averaging.** 1 seed for Gridworld, 5 seeds for RandomWorld. Paper: 50 Gridworld batches, 20 worlds × 5 batches = 100 RandomWorld runs.

8. **"Structured RandomWorld" was an invention.** Not in the paper.

9. **Deterministic optimal expert.** Paper's main results use an ε-optimal expert with ~40% stochastic-policy states.

The v2 (current) results address items 1–8; item 9 is controlled via the `stochastic_fraction` argument in the new expert.
