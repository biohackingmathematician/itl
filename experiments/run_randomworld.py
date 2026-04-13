"""
RandomWorld reproduction experiment.

Reproduces Table 1 (RandomWorld columns) from Benac et al. (2024).

KEY FINDING: ITL's advantage over MLE depends on the true dynamics being
far from uniform. For Dirichlet(1,...,1) random MDPs, the uniform MLE for
unvisited pairs is already near-optimal, so ITL constraints add minimal value.
ITL shines when transitions are CONCENTRATED (sparse), as in structured environments.

This script tests both:
  1. Standard random MDPs (showing ITL ≈ MLE)
  2. Sparse random MDPs (showing ITL > MLE)
  3. "Structured random" MDPs with concentrated transitions

Setup:
  - 15 states, 5 actions
  - gamma = 0.9
  - N = 10 samples per visited (s, a) pair
  - Multiple random seeds for averaging
"""

import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_randomworld
from src.mdp import TabularMDP, deterministic_policy
from src.expert import generate_batch_data, make_stochastic_expert
from src.itl_solver import solve_itl
from src.utils import (
    transition_mse_visited_vs_unvisited,
    print_results_table,
)


def make_structured_randomworld(n_states=15, n_actions=5, gamma=0.9,
                                 sparsity=3, seed=42):
    """
    Random MDP with CONCENTRATED transitions — each (s,a) transitions
    mainly to `sparsity` states with high probability.

    This better represents real-world dynamics (e.g., clinical data)
    where transitions are not uniformly spread across all states.
    """
    rng = np.random.default_rng(seed)
    T = np.zeros((n_states, n_actions, n_states))

    for s in range(n_states):
        for a in range(n_actions):
            # Pick `sparsity` dominant next-states
            dominant = rng.choice(n_states, size=sparsity, replace=False)
            probs = rng.dirichlet(np.ones(sparsity) * 2.0)
            # Assign 95% of mass to dominant states, 5% spread uniformly
            T[s, a, dominant] = 0.95 * probs
            T[s, a] += 0.05 / n_states
            T[s, a] /= T[s, a].sum()

    R = rng.standard_normal((n_states, n_actions))
    return TabularMDP(n_states, n_actions, T, R, gamma)


def run_standard_randomworld():
    """Standard Dirichlet random MDPs (baseline, ITL ≈ MLE)."""
    print("\n--- Standard Random MDP (Dirichlet alpha=1.0) ---")
    print("Expected: ITL ≈ MLE (uniform MLE is already near-optimal)")

    mdp_seeds = [42, 123, 456, 789, 1024]
    epsilons = [0.5, 1.0, 3.0, 5.0, 7.5, 10.0]

    all_results = {eps: {"mle": [], "itl": []} for eps in epsilons}

    for mdp_seed in mdp_seeds:
        mdp = make_randomworld(n_states=15, n_actions=5, gamma=0.9,
                               dirichlet_alpha=1.0, seed=mdp_seed)
        _, Q_star, _ = mdp.compute_optimal_policy()
        optimal_actions = Q_star.argmax(axis=1)
        pi_expert = deterministic_policy(mdp.n_states, mdp.n_actions, optimal_actions)
        N, T_mle = generate_batch_data(mdp, pi_expert, n_samples_per_sa=10, seed=42)

        mle_results = transition_mse_visited_vs_unvisited(mdp.T, T_mle, N)

        for eps in epsilons:
            T_hat, _ = solve_itl(N, T_mle, mdp.R, mdp.gamma, epsilon=eps,
                                 max_iter=10, verbose=False)
            itl_results = transition_mse_visited_vs_unvisited(mdp.T, T_hat, N)
            all_results[eps]["mle"].append(mle_results["mse_all"])
            all_results[eps]["itl"].append(itl_results["mse_all"])

    _print_summary(epsilons, all_results)
    return all_results


def run_structured_randomworld():
    """Structured random MDPs with concentrated transitions (ITL should help)."""
    print("\n--- Structured Random MDP (concentrated transitions) ---")
    print("Expected: ITL > MLE (uniform MLE is far from concentrated truth)")

    mdp_seeds = [42, 123, 456, 789, 1024]
    epsilons = [0.5, 1.0, 3.0, 5.0, 7.5, 10.0, 15.0]

    all_results = {eps: {"mle": [], "itl": []} for eps in epsilons}

    for mdp_seed in mdp_seeds:
        mdp = make_structured_randomworld(n_states=15, n_actions=5, gamma=0.9,
                                           sparsity=3, seed=mdp_seed)
        _, Q_star, _ = mdp.compute_optimal_policy()
        optimal_actions = Q_star.argmax(axis=1)
        pi_expert = deterministic_policy(mdp.n_states, mdp.n_actions, optimal_actions)
        N, T_mle = generate_batch_data(mdp, pi_expert, n_samples_per_sa=10, seed=42)

        mle_results = transition_mse_visited_vs_unvisited(mdp.T, T_mle, N)

        for eps in epsilons:
            T_hat, _ = solve_itl(N, T_mle, mdp.R, mdp.gamma, epsilon=eps,
                                 max_iter=10, verbose=False)
            itl_results = transition_mse_visited_vs_unvisited(mdp.T, T_hat, N)
            all_results[eps]["mle"].append(mle_results["mse_all"])
            all_results[eps]["itl"].append(itl_results["mse_all"])

    _print_summary(epsilons, all_results)
    return all_results


def run_stochastic_expert_randomworld():
    """Random MDPs with stochastic expert (more visited pairs)."""
    print("\n--- Random MDP with Stochastic Expert ---")
    print("Expert uses uniform over epsilon-ball (more coverage)")

    mdp_seeds = [42, 123, 456, 789, 1024]
    expert_eps = 3.0
    epsilons = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0]

    all_results = {eps: {"mle": [], "itl": []} for eps in epsilons}

    for mdp_seed in mdp_seeds:
        mdp = make_structured_randomworld(n_states=15, n_actions=5, gamma=0.9,
                                           sparsity=3, seed=mdp_seed)
        pi_expert = make_stochastic_expert(mdp, expert_eps)
        N, T_mle = generate_batch_data(mdp, pi_expert, n_samples_per_sa=10, seed=42)

        mle_results = transition_mse_visited_vs_unvisited(mdp.T, T_mle, N)

        for eps in epsilons:
            T_hat, _ = solve_itl(N, T_mle, mdp.R, mdp.gamma, epsilon=eps,
                                 max_iter=10, verbose=False)
            itl_results = transition_mse_visited_vs_unvisited(mdp.T, T_hat, N)
            all_results[eps]["mle"].append(mle_results["mse_all"])
            all_results[eps]["itl"].append(itl_results["mse_all"])

    _print_summary(epsilons, all_results)
    return all_results


def _print_summary(epsilons, all_results):
    """Print formatted results table."""
    print(f"\n  {'Epsilon':>10}  {'MLE MSE':>12}  {'ITL MSE':>12}  {'Improvement':>12}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")

    best_eps = None
    best_improvement = -np.inf

    for eps in epsilons:
        mle_mean = np.mean(all_results[eps]["mle"])
        itl_mean = np.mean(all_results[eps]["itl"])
        improvement = (mle_mean - itl_mean) / mle_mean * 100
        print(f"  {eps:>10.1f}  {mle_mean:>12.6f}  {itl_mean:>12.6f}  {improvement:>11.2f}%")

        if improvement > best_improvement:
            best_improvement = improvement
            best_eps = eps

    print(f"\n  Best: epsilon={best_eps} ({best_improvement:.2f}% improvement)")


def main():
    print("=" * 60)
    print("  RANDOMWORLD EXPERIMENTS (15 states, 5 actions)")
    print("  Benac et al. (2024) Reproduction")
    print("=" * 60)

    results_standard = run_standard_randomworld()
    results_structured = run_structured_randomworld()
    results_stochastic = run_stochastic_expert_randomworld()

    # Save all results
    os.makedirs("results/tables", exist_ok=True)

    summary = {
        "standard": {str(k): {"mle": float(np.mean(v["mle"])), "itl": float(np.mean(v["itl"]))}
                     for k, v in results_standard.items()},
        "structured": {str(k): {"mle": float(np.mean(v["mle"])), "itl": float(np.mean(v["itl"]))}
                       for k, v in results_structured.items()},
        "stochastic_expert": {str(k): {"mle": float(np.mean(v["mle"])), "itl": float(np.mean(v["itl"]))}
                              for k, v in results_stochastic.items()},
    }
    with open("results/tables/randomworld_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  KEY INSIGHT FOR THESIS:")
    print("  ITL's advantage requires structured/concentrated dynamics.")
    print("  For uniform-like transitions, MLE is already near-optimal.")
    print("  This motivates C-ITL: additional constraints add value when")
    print("  the environment has exploitable structure.")
    print("=" * 60)
    print("\nDone! Results saved to results/tables/randomworld_results.json")


if __name__ == "__main__":
    main()
