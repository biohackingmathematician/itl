"""
BITL (Bayesian ITL) experiment script — Algorithm 3 from Benac et al. (2024).

Tests Bayesian posterior sampling on:
  1. Corridor (3-state) — fast sanity check
  2. Gridworld (5x5) — main benchmark with uncertainty quantification
  3. RandomWorld (15 states) — outlier detection demonstration

Key outputs:
  - Posterior mean vs MLE vs ITL point estimate comparison
  - 95% credible interval coverage
  - Bayesian regret (posterior disagreement on optimal policy)
  - Outlier trajectory detection via posterior predictive
"""

import numpy as np
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_corridor, make_gridworld, make_randomworld
from src.mdp import TabularMDP, deterministic_policy
from src.expert import generate_batch_data, generate_expert_trajectories, make_stochastic_expert
from src.itl_solver import solve_itl
from src.bitl import (
    bitl_sample,
    detect_outlier_trajectories,
    compute_bayesian_regret,
    posterior_summary,
    posterior_mse,
)
from src.utils import transition_mse_visited_vs_unvisited, print_results_table


def run_corridor_bitl():
    """Quick sanity check: BITL on 3-state corridor."""
    print("\n" + "=" * 60)
    print("  BITL on CORRIDOR (3 states, 2 actions)")
    print("=" * 60)

    mdp = make_corridor(gamma=0.9)
    v_star, Q_star, pi_star = mdp.compute_optimal_policy()
    pi_expert = pi_star

    N, T_mle = generate_batch_data(mdp, pi_expert, n_samples_per_sa=20, seed=42)

    # ITL point estimate (also serves as feasible init)
    T_itl, _ = solve_itl(N, T_mle, mdp.R, mdp.gamma, epsilon=1.0, max_iter=10)

    print(f"  v* = {v_star}")
    print(f"  Running BITL with n_samples=300, n_warmup=150...")

    t0 = time.time()
    samples, info = bitl_sample(
        N, T_mle, mdp.R, mdp.gamma, epsilon=1.0,
        n_samples=300, n_warmup=150,
        step_size=0.01, n_leapfrog=10,
        T_init=T_itl,
        seed=123, verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Compare estimates
    result = posterior_mse(samples, mdp.T)
    mle_mse = np.mean(np.sum((mdp.T - T_mle) ** 2, axis=2))
    itl_mse = np.mean(np.sum((mdp.T - T_itl) ** 2, axis=2))

    print(f"\n  MSE Comparison:")
    print(f"    MLE:               {mle_mse:.6f}")
    print(f"    ITL (point):       {itl_mse:.6f}")
    print(f"    BITL (post. mean): {result['mse_posterior_mean']:.6f}")
    print(f"    95% CI coverage:   {result['coverage_95']:.3f}")

    return samples, info


def run_gridworld_bitl():
    """Main benchmark: Gridworld BITL with Bayesian regret."""
    print("\n" + "=" * 60)
    print("  BITL on GRIDWORLD (25 states, 4 actions)")
    print("=" * 60)

    mdp = make_gridworld(grid_size=5, gamma=0.9, slip_prob=0.2)
    v_star, Q_star, pi_star = mdp.compute_optimal_policy()

    optimal_actions = Q_star.argmax(axis=1)
    pi_expert = deterministic_policy(mdp.n_states, mdp.n_actions, optimal_actions)

    N, T_mle = generate_batch_data(mdp, pi_expert, n_samples_per_sa=10, seed=42)

    # ITL point estimate
    print("  Computing ITL point estimate...")
    T_itl, itl_info = solve_itl(N, T_mle, mdp.R, mdp.gamma, epsilon=7.5, max_iter=10)
    mle_results = transition_mse_visited_vs_unvisited(mdp.T, T_mle, N)
    itl_results = transition_mse_visited_vs_unvisited(mdp.T, T_itl, N)

    print(f"  MLE MSE (all): {mle_results['mse_all']:.6f}")
    print(f"  ITL MSE (all): {itl_results['mse_all']:.6f}")
    print(f"  ITL improvement: {(mle_results['mse_all']-itl_results['mse_all'])/mle_results['mse_all']*100:.2f}%")

    # BITL posterior
    print(f"\n  Running BITL (n_samples=300, n_warmup=150)...")
    t0 = time.time()
    samples, info = bitl_sample(
        N, T_mle, mdp.R, mdp.gamma, epsilon=7.5,
        n_samples=300, n_warmup=150,
        step_size=0.005, n_leapfrog=10,
        barrier_strength=0.05,
        T_init=T_itl,
        seed=42, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    result = posterior_mse(samples, mdp.T)
    print(f"\n  BITL posterior mean MSE: {result['mse_posterior_mean']:.6f}")
    print(f"  95% CI coverage:         {result['coverage_95']:.3f}")

    # Bayesian regret
    print("\n  Computing Bayesian regret (subsample for speed)...")
    subsample = samples[::3]  # every 3rd sample
    regret = compute_bayesian_regret(subsample, mdp.R, mdp.gamma)
    print(f"  Mean Bayesian regret: {regret['mean_regret']:.4f}")
    print(f"  Mean policy disagreement: {regret['mean_disagreement']:.3f}")

    high_regret_states = np.argsort(regret['regret_per_state'])[-5:]
    print(f"  Top-5 uncertain states: {high_regret_states}")
    print(f"    (These are states where more data would help most)")

    return samples, info, regret


def run_outlier_detection_demo():
    """Demonstrate outlier trajectory detection on structured random MDP."""
    print("\n" + "=" * 60)
    print("  BITL OUTLIER DETECTION DEMO")
    print("=" * 60)

    # Use a small structured random world
    from experiments.run_randomworld import make_structured_randomworld

    mdp = make_structured_randomworld(n_states=10, n_actions=4, gamma=0.9, sparsity=3, seed=42)
    _, Q_star, _ = mdp.compute_optimal_policy()
    optimal_actions = Q_star.argmax(axis=1)
    pi_expert = deterministic_policy(mdp.n_states, mdp.n_actions, optimal_actions)

    N, T_mle = generate_batch_data(mdp, pi_expert, n_samples_per_sa=15, seed=42)
    T_itl, _ = solve_itl(N, T_mle, mdp.R, mdp.gamma, epsilon=5.0, max_iter=10)

    # Generate normal trajectories from true MDP
    normal_trajs = generate_expert_trajectories(
        mdp, pi_expert, n_trajectories=40, max_steps=15, seed=0
    )

    # Generate outlier trajectories from a DIFFERENT MDP (corrupted dynamics)
    mdp_corrupt = make_structured_randomworld(n_states=10, n_actions=4, gamma=0.9,
                                               sparsity=3, seed=999)
    outlier_trajs = generate_expert_trajectories(
        mdp_corrupt, pi_expert, n_trajectories=5, max_steps=15, seed=1
    )

    all_trajs = normal_trajs + outlier_trajs
    true_labels = np.array([False] * 40 + [True] * 5)
    print(f"  {len(normal_trajs)} normal + {len(outlier_trajs)} outlier trajectories")

    # Run BITL
    print("  Running BITL sampling...")
    samples, info = bitl_sample(
        N, T_mle, mdp.R, mdp.gamma, epsilon=5.0,
        n_samples=200, n_warmup=100,
        step_size=0.008, n_leapfrog=10,
        T_init=T_itl,
        seed=42, verbose=True,
    )

    # Outlier detection
    print("\n  Running outlier detection...")
    outlier_result = detect_outlier_trajectories(
        all_trajs, samples, threshold_percentile=12.0
    )

    detected = set(outlier_result["outlier_indices"])
    true_set = set(np.where(true_labels)[0])

    tp = len(detected & true_set)
    fp = len(detected - true_set)
    fn = len(true_set - detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Results:")
    print(f"    True outlier indices:    {sorted(true_set)}")
    print(f"    Detected outlier indices: {sorted(detected)}")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall:    {recall:.3f}")
    print(f"    F1:        {f1:.3f}")

    return outlier_result


def main():
    print("=" * 60)
    print("  BAYESIAN ITL (BITL) — Algorithm 3 Experiments")
    print("  Benac et al. (2024) Reproduction")
    print("=" * 60)

    # 1. Quick corridor test
    run_corridor_bitl()

    # 2. Main gridworld benchmark
    run_gridworld_bitl()

    # 3. Outlier detection
    run_outlier_detection_demo()

    print("\n" + "=" * 60)
    print("  ALL BITL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
