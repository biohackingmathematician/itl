# Paper ↔ Textbook ↔ Code mapping

Cross-references between Benac et al. (2024) "Inverse Transition Learning"
(AAAI 2025) and Krause & Hübotter, *Probabilistic Artificial Intelligence*
(Chapters 10–12). Generated 2026-04-13 as part of the ITL reproduction.

All code-location columns point into this repository.

## Notation reminder

| Symbol | Meaning |
|---|---|
| `T[s, a, s']` | transition probability P(s' \| s, a) |
| `R[s, a]` | immediate reward |
| `gamma` | discount factor |
| `v^pi`, `V^pi` | state-value function under policy pi |
| `Q*`, `Q^pi` | state-action value function |
| `P^pi[s, s']` | `sum_a pi(a|s) T(s'|s,a)` |
| `epsilon`-ball | `E_eps(s; T) = { a : max_{a'} Q*(s,a') - Q*(s,a) <= eps }` |

## Core mappings

| Concept / Paper equation | Krause & Hübotter (Ch 10–12) | Code location |
|---|---|---|
| Bellman optimality, `Q*(s,a) = R(s,a) + gamma * E_{s'}[V*(s')]` | Ch 10 Eq 10.9 / Def 10.7 | `src/mdp.py::TabularMDP.compute_q_values` (L71–81) |
| Policy evaluation linear system, `v^pi = (I - gamma P^pi)^{-1} r^pi` | Ch 10 Eq 10.20 (with `P^pi, r^pi` from Eq 10.18) | `src/mdp.py::TabularMDP.compute_value_function` (L61–69); `src/itl_solver.py::compute_linearized_value` (L49–67) |
| Value iteration for `V*`, `pi*` | Ch 10 (value iteration / Bellman optimality fixed point) | `src/mdp.py::TabularMDP.compute_optimal_policy` (L83–106) |
| `eps`-ball as "near-optimal actions under `Q*`" (Def 1) | Ch 10 eps-greedy / Q\*-gap characterizations | `src/mdp.py::TabularMDP.compute_epsilon_ball` (L108–122) |
| ITL constraint (Eq 8/9): `R(s,a) - R(s,a') + gamma (T_sa - T_sa')^T v_lin` bounds | Ch 10 Q-function linearity in `T` | `src/itl_solver.py::_solve_qp` (constraints block L307–329) |
| Linearization trick (page 4): substitute `T_MLE^pi` for `T` in `(I - gamma T^pi)^{-1}` | Ch 10 Eq 10.20 with a fixed policy-evaluation matrix | `src/itl_solver.py::compute_linearized_value` (L49–72) |
| Alternating scheme (Algorithm 1) | Ch 10 policy iteration (same fixed-point structure, swapped roles) | `src/itl_solver.py::solve_itl` main loop (L184–241) |
| Laplace-smoothed MLE, `T_MLE = (N + delta) / sum (N + delta)` (paper Eq 5) | Ch 11 Dirichlet-Categorical conjugacy: posterior mean of `Dir(delta)` prior with counts `N` | `src/expert.py::compute_mle_transitions` (L26–44) |
| BITL posterior `P_eps(T* \| D) ∝ prod_{s,a} Dir(N_{s,a} + delta) * I[constraints]` | Ch 11 Bayesian MBRL posterior over dynamics (Dirichlet posterior per row) | `src/bitl.py::bitl_sample` (L29–195); `_log_posterior_vec` (L288–315) |
| Hamiltonian Monte Carlo with leapfrog | Ch 12 HMC / leapfrog integrator | `src/bitl.py::_hmc_step_vec` (L202–276) |
| Hamiltonian & Metropolis step | Ch 12 HMC Hamiltonian + acceptance ratio | `src/bitl.py::_hamiltonian_vec` + acceptance check (L272–274) |
| Reflected HMC (bounce off constraint hyperplanes) | Ch 12 HMC on constrained domains | `src/bitl.py::_hmc_step_vec` reflection block (L240–253); `_constraint_normal_phi` (L369–379) |
| Bayesian Regret, `BR(s) = max_i V^(i)(s) - mean_i V^(i)(s)` | Ch 11 Bayesian regret as central MBRL criterion | `src/bitl.py::compute_bayesian_regret` (L568–617) |
| Transfer / CVaR application | Ch 11 risk-sensitive Bayesian MBRL | `src/bitl.py::compute_trajectory_likelihood` (L491–519), `detect_outlier_trajectories` (L522–561) |

## What the book adds that the paper omits

| Missing in paper, filled by Krause & Hübotter | Why it matters |
|---|---|
| Why the `Dir(delta)` prior is the *right* prior for categorical `T(.|s,a)` rows (conjugacy + Laplace rule of succession) | Justifies using Eq 5 rather than raw MLE even when `N > 0` |
| Explicit derivation of `v^pi = (I - gamma P^pi)^{-1} r^pi` and convergence conditions | Justifies the linearization trick: the matrix inversion in Eq 10 is actually policy evaluation with a fixed policy |
| Leapfrog order of accuracy, step-size adaptation, and why rejection dominates near boundaries | Drives the reflected-HMC choice in BITL: rejection rate would collapse near the eps-ball surface |
| Bayesian regret vs. frequentist regret framing | Ties BITL's transfer-task reported metric to a textbook criterion |

## What the paper adds that the book doesn't

| Novel in paper (not in Ch 10–12) | Book analogue |
|---|---|
| Identifiability of `T*` from eps-optimal offline data alone | Closest book topic: offline MBRL (partial coverage) but without eps-optimality framing |
| Alternating QP (Algorithm 1) | Structurally similar to Ch 10 policy iteration, but over dynamics not policy |
| Eq-10 quadratic penalty weighted by `N_{s,a,s'}` | Not in Ch 11; paper-specific calibration |
| Constrained-simplex HMC with eps-ball inequalities as active constraints | Ch 12 covers HMC but not this specific constraint set |

## Files containing inline `Ch N` cross-references

- `src/mdp.py` — Ch 10 Eq 10.9, 10.18, 10.20, Def 10.7
- `src/itl_solver.py` — Ch 10 Eq 10.9, 10.20; Ch 11 context for the MLE target
- `src/bitl.py` — Ch 10 Eq 10.9; Ch 11 Dirichlet-Categorical; Ch 12 HMC / leapfrog / reflected HMC
- `src/expert.py` — Ch 10 Eq 10.9; Ch 11 Dirichlet-Categorical conjugacy
- `src/utils.py` — Ch 10 Eq 10.9, 10.20; Ch 11 Bayesian regret

## Caveat

Equation numbers cited as "Eq 10.18 / 10.20 / 10.9 / Def 10.7" have been
carried forward from `src/mdp.py` where they were originally tagged during
the Ch 10 read. Other chapter-level anchors (Ch 11 Dirichlet-Categorical,
Ch 12 HMC / leapfrog / reflected HMC) are conceptual cross-references: the
material is there, but exact equation numbers should be verified against
the print copy before this table is cited outside the repo. If you find a
mismatch, edit this file and the corresponding inline docstring in `src/`
— both are the source of truth.
