"""
Stress test for Combined ITL+IRL on the 5x5 gridworld.

Extends `run_itl_irl_smoke.py` from the 3-state corridor to the paper's
gridworld benchmark (5x5, slip 0.2, gamma=0.95, soft walls, goal at
top-right with reward +10).

Setup
-----
- Reward features Phi(s, a, k) = 1[k == s * n_actions + a], one-hot over
  (s, a) pairs. d = n_states * n_actions = 100.
- True w (latent ground truth): w_true[s * n_actions + a] = R_true[s, a].
  With one-hot (s, a) features there's a separate weight per (s, a) pair, so
  in principle R is identifiable up to additive constants in directions that
  zero out under the eps-ball constraints. We anchor one component to break
  that ambiguity.
- Anchor: at the goal state s* = (0, grid_size-1), action 0 (Up — any action
  works since the goal is absorbing). Phi[s*, 0] @ w = 10.0.
- Expert: deterministic pi*(T*, R*) (matches the corridor smoke test).
- ITL+IRL parameters: epsilon = 5.0 (paper's gridworld value), L1 weight
  lambda_l1 = 0.01.

Reports (acceptance: best-action match >= 0.7):
- Outer iterations until convergence
- Best-action match: |{s : argmax_a pi*(T_hat, R_hat)[s] == argmax_a pi_star[s]}|/|S|
- T MSE vs T_MLE and vs ITL alone (Algorithm 1 with known R)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_gridworld, gridworld_start_state
from src.expert import generate_batch_data
from src.itl_solver import solve_itl
from src.itl_irl_solver import solve_itl_irl
from src.mdp import TabularMDP
from src.utils import best_matching


def _onehot_sa_features(n_states: int, n_actions: int) -> np.ndarray:
    """
    One-hot features over (s, a) pairs:
        Phi[s, a, k] = 1 if k == s * n_actions + a else 0,
    with k = 0, ..., n_states * n_actions - 1.
    """
    d = n_states * n_actions
    Phi = np.zeros((n_states, n_actions, d))
    for s in range(n_states):
        for a in range(n_actions):
            Phi[s, a, s * n_actions + a] = 1.0
    return Phi


def _t_mse(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """Mean over (s, a) of sum_{s'} (T_a - T_b)^2. Same scale as in MVR."""
    return float(np.mean(np.sum((T_a - T_b) ** 2, axis=2)))


def main() -> int:
    print("=" * 64)
    print("  ITL+IRL stress test on 5x5 gridworld")
    print("=" * 64)

    grid_size = 5
    epsilon = 5.0
    lambda_l1 = 0.01
    n_samples_per_sa = 10
    seed = 42

    mdp = make_gridworld(grid_size=grid_size)
    n_states = mdp.n_states
    n_actions = mdp.n_actions
    goal_state = 0 * grid_size + (grid_size - 1)
    start_state = gridworld_start_state(grid_size)

    print(f"  States: {n_states}  Actions: {n_actions}  gamma: {mdp.gamma}")
    print(f"  Goal state: {goal_state} (top-right)   "
          f"Start state: {start_state} (bottom-left)")

    # Optimal policy under the true MDP
    v_star, Q_star, pi_star = mdp.compute_optimal_policy()

    # One-hot (s, a) features
    Phi = _onehot_sa_features(n_states, n_actions)
    d = Phi.shape[2]
    print(f"  Phi shape: {Phi.shape}  (d = {d})")

    # Generate expert dataset under the deterministic optimal policy
    N, T_mle = generate_batch_data(
        mdp, pi_star, n_samples_per_sa=n_samples_per_sa, seed=seed,
    )
    sa_visited = (N.sum(axis=2) > 0).sum()
    print(f"  Total transitions: {int(N.sum())}   "
          f"Visited (s, a) pairs: {sa_visited} / {n_states * n_actions}")

    # ------------------------------------------------------------------
    # ITL alone: known R, learn T only (paper Algorithm 1)
    # ------------------------------------------------------------------
    print("\n  Running ITL alone (known R) ...")
    T_itl, itl_info = solve_itl(
        N, T_mle, mdp.R, mdp.gamma, epsilon=epsilon, max_iter=10, verbose=False,
    )

    # ------------------------------------------------------------------
    # ITL+IRL: learn (T, w) jointly with anchored reward features
    # ------------------------------------------------------------------
    anchor = (goal_state, 0, 10.0)
    print(f"\n  Running ITL+IRL with anchor R(s={goal_state}, a=0) = 10.0, "
          f"epsilon = {epsilon}, lambda_l1 = {lambda_l1} ...")
    T_hat, w_hat, info = solve_itl_irl(
        N, T_mle, Phi, mdp.gamma, epsilon=epsilon,
        anchor=anchor, lambda_l1=lambda_l1,
        max_iter=15, tol=1e-6, verbose=True,
    )

    # ------------------------------------------------------------------
    # Compare hard pi* under each estimate
    # ------------------------------------------------------------------
    R_hat = Phi @ w_hat
    mdp_hat = TabularMDP(n_states, n_actions, T_hat, R_hat, mdp.gamma)
    _, _, pi_itl_irl = mdp_hat.compute_optimal_policy()

    mdp_itl = TabularMDP(n_states, n_actions, T_itl, mdp.R, mdp.gamma)
    _, _, pi_itl = mdp_itl.compute_optimal_policy()

    mdp_mle = TabularMDP(n_states, n_actions, T_mle, mdp.R, mdp.gamma)
    _, _, pi_mle = mdp_mle.compute_optimal_policy()

    match_itl_irl = best_matching(pi_itl_irl, pi_star)
    match_itl = best_matching(pi_itl, pi_star)
    match_mle = best_matching(pi_mle, pi_star)

    # ------------------------------------------------------------------
    # T errors
    # ------------------------------------------------------------------
    mse_mle = _t_mse(mdp.T, T_mle)
    mse_itl = _t_mse(mdp.T, T_itl)
    mse_itl_irl = _t_mse(mdp.T, T_hat)
    mse_itl_irl_vs_mle = _t_mse(T_hat, T_mle)
    mse_itl_irl_vs_itl = _t_mse(T_hat, T_itl)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    n_outer = len(info["iterations"])
    print("\n" + "-" * 64)
    print(f"  Outer iterations:    {n_outer}")
    print(f"  Termination:         {info['termination']}")
    print(f"  Converged:           {info.get('converged', False)}")
    print()
    print("  Best-action match (vs pi*(T*, R*)):")
    print(f"    MLE      : {match_mle:.3f}")
    print(f"    ITL      : {match_itl:.3f}")
    print(f"    ITL+IRL  : {match_itl_irl:.3f}")
    print()
    print("  T MSE  (mean over (s,a) of sum_{s'} squared error):")
    print(f"    T_MLE   vs T*     : {mse_mle:.4f}")
    print(f"    T_ITL   vs T*     : {mse_itl:.4f}")
    print(f"    T_HAT   vs T*     : {mse_itl_irl:.4f}")
    print(f"    T_HAT   vs T_MLE  : {mse_itl_irl_vs_mle:.4f}")
    print(f"    T_HAT   vs T_ITL  : {mse_itl_irl_vs_itl:.4f}")

    accept = match_itl_irl >= 0.7
    print()
    print("=" * 64)
    print(f"  ACCEPTANCE (match >= 0.70): "
          f"{'PASS' if accept else 'FAIL'}  (got {match_itl_irl:.3f})")
    print("=" * 64)

    return 0 if accept else 1


if __name__ == "__main__":
    sys.exit(main())
