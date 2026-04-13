"""
Gridworld reproduction — Benac et al. (2024), Table 4 (Gridworld, 40% stochastic
expert, standard task) and Figure 2 (Normalized Value vs. Coverage).

This is the MVR (minimum viable reproduction) version: fewer seeds than the
paper's 50 to keep runtime manageable. Error bars will be wider than published.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_gridworld, gridworld_start_state
from src.expert import make_epsilon_optimal_expert, generate_batch_dataset
from src.itl_solver import solve_itl
from src.utils import (
    best_matching,
    epsilon_matching,
    normalized_value,
    total_variation,
    count_constraint_violations,
    transition_mse,
    summarize_runs,
    plot_coverage_sensitivity,
)
from src.mdp import TabularMDP


def evaluate(T_hat, R, gamma, true_mdp, epsilon, start_state):
    """Compute all paper metrics for a single learned T_hat."""
    mdp_learned = TabularMDP(
        true_mdp.n_states, true_mdp.n_actions, T_hat, R, gamma
    )
    _, _, pi_hat = mdp_learned.compute_optimal_policy()
    _, Q_star, pi_star = true_mdp.compute_optimal_policy()

    return {
        "normalized_value": normalized_value(
            T_hat, R, gamma, true_mdp, start_state=start_state
        ),
        "best_matching": best_matching(pi_hat, pi_star),
        "epsilon_matching": epsilon_matching(pi_hat, Q_star, epsilon),
        "total_variation": total_variation(true_mdp.T, T_hat),
        "n_violations": count_constraint_violations(
            T_hat, R, gamma, true_mdp, epsilon
        ),
        "mse": transition_mse(true_mdp.T, T_hat),
    }


def run_one(mdp, pi_expert, coverage, epsilon, K, delta, seed):
    N, T_mle, _ = generate_batch_dataset(
        mdp, pi_expert, coverage=coverage, K=K, delta=delta, seed=seed
    )
    T_hat, _ = solve_itl(
        N, T_mle, mdp.R, mdp.gamma, epsilon=epsilon, max_iter=10, verbose=False
    )
    return T_mle, T_hat, N


def main():
    print("=" * 70)
    print("  GRIDWORLD — Benac et al. (2024) MVR (Table 4 / Figure 2)")
    print("=" * 70)

    # Paper constants
    GAMMA = 0.95
    DELTA = 0.001
    EPSILON = 5.0                 # main ε from paper
    STOCHASTIC_FRACTION = 0.4     # main setup (40% stochastic-policy states)
    K = 10                        # paper's Gridworld K
    COVERAGES = [0.2, 0.4, 0.6, 0.8, 1.0]
    N_SEEDS = int(os.environ.get("N_SEEDS", "10"))  # MVR default 10 (paper uses 50)

    mdp = make_gridworld(
        grid_size=5, gamma=GAMMA, slip_prob=0.2, transfer=False
    )
    print(f"  States: {mdp.n_states}, Actions: {mdp.n_actions}")
    v_star, Q_star, pi_star = mdp.compute_optimal_policy()
    print(f"  V* range: [{v_star.min():.2f}, {v_star.max():.2f}]")

    pi_expert = make_epsilon_optimal_expert(
        mdp,
        epsilon=EPSILON,
        target_stochastic_fraction=STOCHASTIC_FRACTION,
    )
    n_stochastic = int(((pi_expert > 0).sum(axis=1) > 1).sum())
    print(f"  Expert: {n_stochastic}/{mdp.n_states} stochastic-policy states "
          f"(target {STOCHASTIC_FRACTION:.0%})")

    start_state = gridworld_start_state(grid_size=5)
    print(f"  Start state: {start_state} (bottom-left)")

    # --- Coverage sweep with per-seed checkpointing ---
    os.makedirs("results/checkpoints", exist_ok=True)
    ckpt_path = "results/checkpoints/gridworld_seeds.json"
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            checkpoint = json.load(f)
    else:
        checkpoint = {}

    rows = []
    for coverage in COVERAGES:
        mle_runs = {k: [] for k in ["normalized_value", "best_matching",
                                     "epsilon_matching", "total_variation",
                                     "n_violations", "mse"]}
        itl_runs = {k: [] for k in mle_runs}

        for seed in range(N_SEEDS):
            key = f"{coverage}:{seed}"
            if key in checkpoint:
                mle_metrics = checkpoint[key]["mle"]
                itl_metrics = checkpoint[key]["itl"]
            else:
                T_mle, T_hat, _ = run_one(
                    mdp, pi_expert, coverage, EPSILON, K, DELTA, seed=seed
                )
                mle_metrics = evaluate(T_mle, mdp.R, mdp.gamma, mdp, EPSILON, start_state)
                itl_metrics = evaluate(T_hat, mdp.R, mdp.gamma, mdp, EPSILON, start_state)
                checkpoint[key] = {"mle": mle_metrics, "itl": itl_metrics}
                with open(ckpt_path, "w") as f:
                    json.dump(checkpoint, f)
            for k in mle_runs:
                mle_runs[k].append(mle_metrics[k])
                itl_runs[k].append(itl_metrics[k])

        row = {"coverage": coverage}
        for k in mle_runs:
            m_mean, m_std = summarize_runs(mle_runs[k])
            i_mean, i_std = summarize_runs(itl_runs[k])
            row[f"mle_{k}_mean"] = m_mean
            row[f"mle_{k}_std"] = m_std
            row[f"itl_{k}_mean"] = i_mean
            row[f"itl_{k}_std"] = i_std
        rows.append(row)

        print(f"\n  Coverage = {coverage:.1f}  "
              f"(avg over {N_SEEDS} seeds)")
        print(f"    Normalized Value  MLE: {row['mle_normalized_value_mean']:+.3f} "
              f"± {row['mle_normalized_value_std']:.3f}   "
              f"ITL: {row['itl_normalized_value_mean']:+.3f} "
              f"± {row['itl_normalized_value_std']:.3f}")
        print(f"    Best matching     MLE: {row['mle_best_matching_mean']:.3f} "
              f"± {row['mle_best_matching_std']:.3f}   "
              f"ITL: {row['itl_best_matching_mean']:.3f} "
              f"± {row['itl_best_matching_std']:.3f}")
        print(f"    ε-matching        MLE: {row['mle_epsilon_matching_mean']:.3f} "
              f"± {row['mle_epsilon_matching_std']:.3f}   "
              f"ITL: {row['itl_epsilon_matching_mean']:.3f} "
              f"± {row['itl_epsilon_matching_std']:.3f}")
        print(f"    Violations        MLE: {row['mle_n_violations_mean']:.2f}   "
              f"ITL: {row['itl_n_violations_mean']:.2f}")

    # Save
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    out_path = "results/tables/gridworld_coverage_sweep.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "gamma": GAMMA, "delta": DELTA, "epsilon": EPSILON,
                "stochastic_fraction": STOCHASTIC_FRACTION,
                "K": K, "n_seeds": N_SEEDS, "coverages": COVERAGES,
            },
            "rows": rows,
        }, f, indent=2)

    # Plot: Normalized Value vs Coverage
    plot_coverage_sensitivity(
        coverages=[r["coverage"] for r in rows],
        itl_mean=[r["itl_normalized_value_mean"] for r in rows],
        itl_std=[r["itl_normalized_value_std"] for r in rows],
        mle_mean=[r["mle_normalized_value_mean"] for r in rows],
        mle_std=[r["mle_normalized_value_std"] for r in rows],
        metric_name="Normalized Value",
        title=f"Gridworld — ITL vs MLE (ε={EPSILON}, 40% stochastic expert)",
        save_path="results/figures/gridworld_normalized_value_vs_coverage.png",
    )
    plot_coverage_sensitivity(
        coverages=[r["coverage"] for r in rows],
        itl_mean=[r["itl_best_matching_mean"] for r in rows],
        itl_std=[r["itl_best_matching_std"] for r in rows],
        mle_mean=[r["mle_best_matching_mean"] for r in rows],
        mle_std=[r["mle_best_matching_std"] for r in rows],
        metric_name="Best matching",
        title="Gridworld — Best matching vs Coverage",
        save_path="results/figures/gridworld_best_matching_vs_coverage.png",
    )

    print(f"\n  Results written to {out_path}")
    print("  Figures in results/figures/")


if __name__ == "__main__":
    main()
