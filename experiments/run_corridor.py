"""
Verification experiment: 3-state corridor from Agna's hand calculations.

This script verifies the ITL solver produces results consistent with
the hand-worked example in ITL_paper_notes (March 2026).

Expected values (gamma=0.9, epsilon=1.0):
  v* = [76.87, 87.68, 100.00]
  Q*(s1, L) = 69.08, Q*(s1, R) = 76.87
  Q*(s2, L) = 71.03, Q*(s2, R) = 87.68
  Constraint 1 at s1: 7.78 >= 1.0  (check)
  Constraint 1 at s2: 16.65 >= 1.0 (check)
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_corridor
from src.mdp import deterministic_policy
from src.expert import generate_batch_data
from src.itl_solver import solve_itl
from src.utils import transition_mse_visited_vs_unvisited, print_results_table


def main():
    print("=" * 60)
    print("  CORRIDOR VERIFICATION (3-state toy MDP)")
    print("  Checking against hand calculations from notes")
    print("=" * 60)

    # ---- Step 1: Build environment and verify values ----
    mdp = make_corridor(gamma=0.9)
    v_star, Q_star, pi_star = mdp.compute_optimal_policy()

    print("\n--- Step 1: Value function verification ---")
    print(f"  v*(s1)    = {v_star[0]:.2f}  (expected: 76.87)")
    print(f"  v*(s2)    = {v_star[1]:.2f}  (expected: 87.68)")
    print(f"  v*(sgoal) = {v_star[2]:.2f}  (expected: 100.00)")

    print(f"\n  Q*(s1, L) = {Q_star[0, 0]:.2f}  (expected: 69.08)")
    print(f"  Q*(s1, R) = {Q_star[0, 1]:.2f}  (expected: 76.87)")
    print(f"  Q*(s2, L) = {Q_star[1, 0]:.2f}  (expected: 71.03)")
    print(f"  Q*(s2, R) = {Q_star[1, 1]:.2f}  (expected: 87.68)")

    # Verify epsilon-ball
    epsilon = 1.0
    valid = mdp.compute_epsilon_ball(Q_star, epsilon)
    print(f"\n--- Step 2: Epsilon-ball (epsilon={epsilon}) ---")
    action_names = ["Left", "Right"]
    state_names = ["s1", "s2", "sgoal"]
    for s in range(3):
        for a in range(2):
            gap = Q_star[s].max() - Q_star[s, a]
            status = "VALID" if valid[s, a] else "INVALID"
            print(f"  {state_names[s]}, {action_names[a]}: gap={gap:.2f}, {status}")

    # ---- Step 3: Generate expert data and run ITL ----
    print(f"\n--- Step 3: Generate expert data ---")
    # Expert always chooses Right (action 1)
    pi_expert = deterministic_policy(3, 2, np.array([1, 1, 1]))

    N, T_mle = generate_batch_data(mdp, pi_expert, n_samples_per_sa=20, seed=42)
    print(f"  Total transitions: {N.sum():.0f}")
    print(f"  Visited (s,a) pairs: {(N.sum(axis=2) > 0).sum()}")
    print(f"  Unvisited (s,a) pairs: {(N.sum(axis=2) == 0).sum()}")

    # ---- Step 4: Run ITL ----
    print(f"\n--- Step 4: Run ITL solver ---")
    T_hat, info = solve_itl(
        N, T_mle, mdp.R, mdp.gamma, epsilon=epsilon, max_iter=10, verbose=True
    )

    # ---- Step 5: Compare results ----
    print(f"\n--- Step 5: Results ---")

    # MLE results
    mle_results = transition_mse_visited_vs_unvisited(mdp.T, T_mle, N)
    print_results_table(mle_results, "MLE Baseline")

    # ITL results
    itl_results = transition_mse_visited_vs_unvisited(mdp.T, T_hat, N)
    print_results_table(itl_results, "ITL")

    # Key comparison
    print("--- Key comparison (unvisited pairs) ---")
    print(f"  MLE MSE (unvisited): {mle_results['mse_unvisited']:.6f}")
    print(f"  ITL MSE (unvisited): {itl_results['mse_unvisited']:.6f}")
    if itl_results['mse_unvisited'] < mle_results['mse_unvisited']:
        print("  ITL beats MLE on unvisited pairs!")
    else:
        print("  WARNING: ITL did not beat MLE. Check epsilon or solver convergence.")

    # Print actual vs estimated transitions for unvisited pairs
    print(f"\n--- Estimated transitions for unvisited (Left) actions ---")
    for s in range(2):  # s1 and s2 only
        print(f"  {state_names[s]}, Left:")
        print(f"    True:      {mdp.T[s, 0]}")
        print(f"    MLE:       {T_mle[s, 0]}  (uniform — no data)")
        print(f"    ITL:       {T_hat[s, 0]}")


if __name__ == "__main__":
    main()
