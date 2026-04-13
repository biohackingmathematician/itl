"""
Gridworld reproduction experiment.

Reproduces Table 1 (Gridworld columns) and Figure 1 (epsilon sensitivity)
from Benac et al. (2024).

Setup:
  - 5x5 grid, 25 states, 4 actions (Up/Down/Left/Right)
  - slip_prob = 0.1
  - gamma = 0.9
  - Goal at bottom-right corner
  - Expert is epsilon-optimal (deterministic, always chooses optimal action)
  - N = 10 samples per visited (s, a) pair (paper default)
  - Epsilon sweep: [0.1, 1, 3, 5, 7.5, 10, 12.5, 15]
"""

import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_gridworld
from src.mdp import deterministic_policy
from src.expert import generate_batch_data
from src.itl_solver import solve_itl
from src.utils import (
    transition_mse_visited_vs_unvisited,
    print_results_table,
    plot_epsilon_sensitivity,
)


def run_single_experiment(
    mdp, pi_expert, n_samples: int, epsilon: float, seed: int, verbose: bool = False
):
    """Run ITL and MLE for a single (n_samples, epsilon) setting."""
    N, T_mle = generate_batch_data(mdp, pi_expert, n_samples_per_sa=n_samples, seed=seed)

    # MLE baseline
    mle_results = transition_mse_visited_vs_unvisited(mdp.T, T_mle, N)

    # ITL
    T_hat, info = solve_itl(
        N, T_mle, mdp.R, mdp.gamma, epsilon=epsilon, max_iter=10, verbose=verbose
    )
    itl_results = transition_mse_visited_vs_unvisited(mdp.T, T_hat, N)

    return mle_results, itl_results, info


def main():
    print("=" * 60)
    print("  GRIDWORLD EXPERIMENT (5x5, 25 states)")
    print("  Reproducing Table 1 and Figure 1 from Benac et al.")
    print("=" * 60)

    # Build environment
    mdp = make_gridworld(grid_size=5, gamma=0.9, slip_prob=0.1)
    v_star, Q_star, pi_star = mdp.compute_optimal_policy()

    print(f"\n  States: {mdp.n_states}, Actions: {mdp.n_actions}")
    print(f"  v* range: [{v_star.min():.2f}, {v_star.max():.2f}]")

    # Expert policy (deterministic optimal)
    optimal_actions = Q_star.argmax(axis=1)
    pi_expert = deterministic_policy(mdp.n_states, mdp.n_actions, optimal_actions)

    # --- Epsilon sensitivity sweep (Figure 1) ---
    print("\n--- Epsilon sensitivity sweep ---")
    epsilons = [0.1, 1.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0]
    n_samples = 10
    seed = 42

    mse_itl_all = []
    mse_mle_all = []
    results_table = []

    for eps in epsilons:
        print(f"\n  epsilon = {eps}:")
        mle_res, itl_res, info = run_single_experiment(
            mdp, pi_expert, n_samples, eps, seed, verbose=False
        )

        mse_itl_all.append(itl_res["mse_all"])
        mse_mle_all.append(mle_res["mse_all"])

        print(f"    MLE MSE (all):       {mle_res['mse_all']:.6f}")
        print(f"    ITL MSE (all):       {itl_res['mse_all']:.6f}")
        print(f"    MLE MSE (unvisited): {mle_res['mse_unvisited']:.6f}")
        print(f"    ITL MSE (unvisited): {itl_res['mse_unvisited']:.6f}")
        print(f"    Converged: {info['converged']}")

        results_table.append({
            "epsilon": eps,
            "mle_mse_all": mle_res["mse_all"],
            "itl_mse_all": itl_res["mse_all"],
            "mle_mse_visited": mle_res["mse_visited"],
            "itl_mse_visited": itl_res["mse_visited"],
            "mle_mse_unvisited": mle_res["mse_unvisited"],
            "itl_mse_unvisited": itl_res["mse_unvisited"],
        })

    # Save results
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    with open("results/tables/gridworld_epsilon_sweep.json", "w") as f:
        json.dump(results_table, f, indent=2)

    # Plot
    try:
        plot_epsilon_sensitivity(
            epsilons, mse_itl_all, mse_mle_all,
            title="Gridworld: ITL vs MLE (Epsilon Sensitivity)",
            save_path="results/figures/gridworld_epsilon_sensitivity.png",
        )
    except Exception as e:
        print(f"  (Plotting skipped: {e})")

    # --- Best epsilon detailed results ---
    best_idx = np.argmin(mse_itl_all)
    best_eps = epsilons[best_idx]
    print(f"\n--- Best epsilon: {best_eps} (MSE = {mse_itl_all[best_idx]:.6f}) ---")

    mle_res, itl_res, _ = run_single_experiment(
        mdp, pi_expert, n_samples, best_eps, seed, verbose=True
    )
    print_results_table(mle_res, "MLE Baseline")
    print_results_table(itl_res, f"ITL (epsilon={best_eps})")

    print("\nDone! Results saved to results/tables/ and results/figures/")


if __name__ == "__main__":
    main()
