"""
Smoke test for Combined ITL+IRL on the 3-state corridor.

Verifies the prototype solver in `src/itl_irl_solver.py` recovers a w_hat
such that R_hat = Phi @ w_hat is close to R_true (up to the additive
constant that's only fixed by the anchor).

Setup:
- Corridor MDP with R(s, a) = R(s) only:
  R(0, *) = -0.1, R(1, *) = -0.1, R(2, *) = +10
- One-hot state features: Phi(s, a, k) = 1[s == k]; d = 3.
- True w = [-0.1, -0.1, 10.0].
- Anchor at (s=2, a=0): R(2, 0) = 10.0 → w[2] == 10.0.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_corridor
from src.expert import generate_batch_data
from src.itl_irl_solver import solve_itl_irl


def main():
    print("=" * 60)
    print("  ITL+IRL smoke test on corridor")
    print("=" * 60)

    mdp = make_corridor(gamma=0.9)
    v_star, Q_star, pi_star = mdp.compute_optimal_policy()

    n_states, n_actions = mdp.n_states, mdp.n_actions

    # State-only one-hot features
    d = n_states
    Phi = np.zeros((n_states, n_actions, d))
    for s in range(n_states):
        for a in range(n_actions):
            Phi[s, a, s] = 1.0

    # True reward weights under one-hot features (each state has one weight)
    w_true = np.array([mdp.R[s, 0] for s in range(n_states)])
    print(f"\n  True w (state-rewards): {w_true}")

    # Generate expert data following pi_star
    N, T_mle = generate_batch_data(mdp, pi_star, n_samples_per_sa=20, seed=42)
    print(f"  N total: {int(N.sum())}; visited (s,a): {(N.sum(axis=2) > 0).sum()}")

    # Solve joint ITL+IRL with anchor at (s=2, a=0): R = 10
    print("\n  Solving with anchor R(2, 0) = 10.0, lambda_l1 = 0.01 ...")
    T_hat, w_hat, info = solve_itl_irl(
        N, T_mle, Phi, mdp.gamma, epsilon=1.0,
        anchor=(2, 0, 10.0), lambda_l1=0.01,
        max_iter=10, verbose=True,
    )

    print(f"\n  Termination: {info['termination']}")
    print(f"  Iterations: {len(info['iterations'])}")
    print(f"\n  w_hat:    {w_hat.round(4)}")
    print(f"  w_true:   {w_true.round(4)}")
    print(f"  L_inf err: {np.max(np.abs(w_hat - w_true)):.4f}")

    # Compare T recovery
    T_mse = float(np.mean(np.sum((mdp.T - T_hat) ** 2, axis=2)))
    T_mse_mle = float(np.mean(np.sum((mdp.T - T_mle) ** 2, axis=2)))
    print(f"\n  T MSE  MLE:    {T_mse_mle:.4f}")
    print(f"  T MSE  ITL+IRL: {T_mse:.4f}")

    # Sanity: does pi*(T_hat, R_hat) match pi*(T*, R*) ?
    from src.mdp import TabularMDP
    R_hat = Phi @ w_hat
    mdp_hat = TabularMDP(n_states, n_actions, T_hat, R_hat, mdp.gamma)
    _, _, pi_hat = mdp_hat.compute_optimal_policy()
    match = float(np.mean(pi_hat.argmax(axis=1) == pi_star.argmax(axis=1)))
    print(f"\n  pi* match (best action): {match:.3f}  (1.0 = exact)")


if __name__ == "__main__":
    main()
