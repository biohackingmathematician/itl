"""
RandomWorld reproduction — Benac et al. (2024), Table 4 (Randomworlds, 40%
stochastic expert, standard task) and Figure 2 bottom row.

MVR: 5 worlds × 2 dataset seeds = 10 runs per coverage value.
Paper uses 20 × 5 = 100 runs; error bars here will be wider.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_randomworld
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


def evaluate(T_hat, R, gamma, true_mdp, epsilon):
    """RandomWorld has uniform initial distribution, so no single start state."""
    mdp_learned = TabularMDP(
        true_mdp.n_states, true_mdp.n_actions, T_hat, R, gamma
    )
    _, _, pi_hat = mdp_learned.compute_optimal_policy()
    _, Q_star, pi_star = true_mdp.compute_optimal_policy()

    return {
        "normalized_value": normalized_value(
            T_hat, R, gamma, true_mdp, start_state=None
        ),
        "best_matching": best_matching(pi_hat, pi_star),
        "epsilon_matching": epsilon_matching(pi_hat, Q_star, epsilon),
        "total_variation": total_variation(true_mdp.T, T_hat),
        "n_violations": count_constraint_violations(
            T_hat, R, gamma, true_mdp, epsilon
        ),
        "mse": transition_mse(true_mdp.T, T_hat),
    }


def main():
    print("=" * 70)
    print("  RANDOMWORLD — Benac et al. (2024) MVR (Table 4 / Figure 2 bottom)")
    print("=" * 70)

    # Paper constants
    GAMMA = 0.95
    DELTA = 0.001
    EPSILON = 5.0
    STOCHASTIC_FRACTION = float(os.environ.get("STOCHASTIC_FRACTION", "0.4"))
    K = 5                          # paper's RandomWorld K
    COVERAGES = [0.2, 0.4, 0.6, 0.8, 1.0]
    N_WORLDS = int(os.environ.get("N_WORLDS", "5"))                # paper: 20
    N_DATASETS_PER_WORLD = int(os.environ.get("N_DATASETS", "2"))  # paper: 5

    os.makedirs("results/checkpoints", exist_ok=True)
    sf_tag = f"sf{int(round(STOCHASTIC_FRACTION * 100)):03d}"
    ckpt_path = f"results/checkpoints/randomworld_runs_{sf_tag}.json"
    legacy_path = "results/checkpoints/randomworld_runs.json"
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            checkpoint = json.load(f)
    elif STOCHASTIC_FRACTION == 0.4 and os.path.exists(legacy_path):
        with open(legacy_path) as f:
            checkpoint = json.load(f)
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f)
    else:
        checkpoint = {}

    rows = []
    for coverage in COVERAGES:
        mle_runs = {k: [] for k in ["normalized_value", "best_matching",
                                     "epsilon_matching", "total_variation",
                                     "n_violations", "mse"]}
        itl_runs = {k: [] for k in mle_runs}

        for world_seed in range(N_WORLDS):
            mdp = make_randomworld(
                n_states=15, n_actions=5, gamma=GAMMA,
                n_successors=5, seed=world_seed,
            )
            pi_expert = make_epsilon_optimal_expert(
                mdp, epsilon=EPSILON,
                target_stochastic_fraction=STOCHASTIC_FRACTION,
            )

            for data_seed in range(N_DATASETS_PER_WORLD):
                key = f"{coverage}:{world_seed}:{data_seed}"
                if key in checkpoint:
                    mle_metrics = checkpoint[key]["mle"]
                    itl_metrics = checkpoint[key]["itl"]
                else:
                    combined_seed = 1000 * world_seed + data_seed
                    N, T_mle, _ = generate_batch_dataset(
                        mdp, pi_expert, coverage=coverage, K=K,
                        delta=DELTA, seed=combined_seed,
                    )
                    T_hat, _ = solve_itl(
                        N, T_mle, mdp.R, mdp.gamma,
                        epsilon=EPSILON, max_iter=10, verbose=False,
                    )
                    mle_metrics = evaluate(T_mle, mdp.R, mdp.gamma, mdp, EPSILON)
                    itl_metrics = evaluate(T_hat, mdp.R, mdp.gamma, mdp, EPSILON)
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
              f"(avg over {N_WORLDS * N_DATASETS_PER_WORLD} runs)")
        print(f"    Normalized Value  MLE: {row['mle_normalized_value_mean']:+.3f} "
              f"± {row['mle_normalized_value_std']:.3f}   "
              f"ITL: {row['itl_normalized_value_mean']:+.3f} "
              f"± {row['itl_normalized_value_std']:.3f}")
        print(f"    Best matching     MLE: {row['mle_best_matching_mean']:.3f} "
              f"± {row['mle_best_matching_std']:.3f}   "
              f"ITL: {row['itl_best_matching_mean']:.3f} "
              f"± {row['itl_best_matching_std']:.3f}")
        print(f"    ε-matching        MLE: {row['mle_epsilon_matching_mean']:.3f}   "
              f"ITL: {row['itl_epsilon_matching_mean']:.3f}")

    # Save
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    out_path = f"results/tables/randomworld_coverage_sweep_{sf_tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "gamma": GAMMA, "delta": DELTA, "epsilon": EPSILON,
                "stochastic_fraction": STOCHASTIC_FRACTION, "K": K,
                "n_worlds": N_WORLDS, "n_datasets_per_world": N_DATASETS_PER_WORLD,
                "coverages": COVERAGES,
            },
            "rows": rows,
        }, f, indent=2)

    sf_pct = int(round(STOCHASTIC_FRACTION * 100))
    plot_coverage_sensitivity(
        coverages=[r["coverage"] for r in rows],
        itl_mean=[r["itl_normalized_value_mean"] for r in rows],
        itl_std=[r["itl_normalized_value_std"] for r in rows],
        mle_mean=[r["mle_normalized_value_mean"] for r in rows],
        mle_std=[r["mle_normalized_value_std"] for r in rows],
        metric_name="Normalized Value",
        title=f"RandomWorld — ITL vs MLE (ε={EPSILON}, {sf_pct}% stochastic expert)",
        save_path=f"results/figures/randomworld_normalized_value_vs_coverage_{sf_tag}.png",
    )
    plot_coverage_sensitivity(
        coverages=[r["coverage"] for r in rows],
        itl_mean=[r["itl_best_matching_mean"] for r in rows],
        itl_std=[r["itl_best_matching_std"] for r in rows],
        mle_mean=[r["mle_best_matching_mean"] for r in rows],
        mle_std=[r["mle_best_matching_std"] for r in rows],
        metric_name="Best matching",
        title=f"RandomWorld — Best matching vs Coverage ({sf_pct}% stochastic)",
        save_path=f"results/figures/randomworld_best_matching_vs_coverage_{sf_tag}.png",
    )

    print(f"\n  Results written to {out_path}")
    print("  Figures in results/figures/")


if __name__ == "__main__":
    main()
