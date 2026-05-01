"""
Transfer task driver — Benac et al. (2024) Table 4 transfer-task columns.

Setup (per paper Section 5):
  1. Generate batch data D under the STANDARD task (R_standard).
  2. Learn T_hat from D (ITL / MLE / PS / etc.).
  3. EVALUATE T_hat on the TRANSFER task (R_transfer): different reward,
     same dynamics. Optimal policy under (T_hat, R_transfer) is then
     scored against the optimal policy under (T*, R_transfer).

Why this matters: transfer is the only test that separates "I learned a
policy" from "I learned the dynamics". A method that overfits to the
training-task reward will collapse on the transfer task; ITL should not.

Both gridworld and randomworld are supported.

Env vars:
  ENV                ∈ {"gridworld", "randomworld"}, default "gridworld"
  STOCHASTIC_FRACTION ∈ float, default 0.4
  N_SEEDS            for gridworld, default 10
  N_WORLDS, N_DATASETS for randomworld, defaults 5, 2
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import (
    make_gridworld, gridworld_start_state,
    make_randomworld, make_randomworld_transfer,
)
from src.expert import make_epsilon_optimal_expert, generate_batch_dataset
from src.itl_solver import solve_itl
from src.utils import (
    best_matching, epsilon_matching, normalized_value, total_variation,
    count_constraint_violations, transition_mse, summarize_runs,
)
from src.mdp import TabularMDP


# Paper constants (shared with run_gridworld / run_randomworld).
GAMMA = 0.95
DELTA = 0.001
EPSILON = 5.0
COVERAGES = [0.2, 0.4, 0.6, 0.8, 1.0]


def _evaluate_transfer(T_hat, true_mdp_transfer, epsilon, start_state):
    """Score T_hat under the transfer-task MDP (transfer reward, true dynamics)."""
    n_states, n_actions = true_mdp_transfer.n_states, true_mdp_transfer.n_actions

    mdp_learned = TabularMDP(n_states, n_actions, T_hat, true_mdp_transfer.R, true_mdp_transfer.gamma)
    _, _, pi_hat = mdp_learned.compute_optimal_policy()
    _, Q_star_t, pi_star_t = true_mdp_transfer.compute_optimal_policy()

    return {
        "normalized_value_transfer": normalized_value(
            T_hat, true_mdp_transfer.R, true_mdp_transfer.gamma,
            true_mdp_transfer, start_state=start_state,
        ),
        "best_matching_transfer": best_matching(pi_hat, pi_star_t),
        "epsilon_matching_transfer": epsilon_matching(pi_hat, Q_star_t, epsilon),
        "n_violations_transfer": count_constraint_violations(
            T_hat, true_mdp_transfer.R, true_mdp_transfer.gamma,
            true_mdp_transfer, epsilon,
        ),
    }


# ---------------------------------------------------------------------------
# Gridworld transfer
# ---------------------------------------------------------------------------

def run_gridworld_transfer(stochastic_fraction, n_seeds):
    print("=" * 70)
    print(f"  GRIDWORLD TRANSFER  (stochastic = {stochastic_fraction:.0%})")
    print("=" * 70)

    mdp_std = make_gridworld(grid_size=5, gamma=GAMMA, slip_prob=0.2, transfer=False)
    mdp_trn = make_gridworld(grid_size=5, gamma=GAMMA, slip_prob=0.2, transfer=True)
    pi_expert = make_epsilon_optimal_expert(
        mdp_std, epsilon=EPSILON, target_stochastic_fraction=stochastic_fraction,
    )
    start_state = gridworld_start_state(grid_size=5)

    sf_tag = f"sf{int(round(stochastic_fraction * 100)):03d}"
    ckpt_path = f"results/checkpoints/gridworld_transfer_{sf_tag}.json"
    os.makedirs("results/checkpoints", exist_ok=True)
    checkpoint = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

    rows = []
    for coverage in COVERAGES:
        mle_runs = {k: [] for k in ["normalized_value_transfer", "best_matching_transfer",
                                     "epsilon_matching_transfer", "n_violations_transfer"]}
        itl_runs = {k: [] for k in mle_runs}
        for seed in range(n_seeds):
            key = f"{coverage}:{seed}"
            if key in checkpoint:
                m = checkpoint[key]["mle"]; i = checkpoint[key]["itl"]
            else:
                N, T_mle, _ = generate_batch_dataset(
                    mdp_std, pi_expert, coverage=coverage, K=10, delta=DELTA, seed=seed,
                )
                T_hat, _ = solve_itl(N, T_mle, mdp_std.R, mdp_std.gamma,
                                     epsilon=EPSILON, max_iter=10, verbose=False)
                m = _evaluate_transfer(T_mle, mdp_trn, EPSILON, start_state)
                i = _evaluate_transfer(T_hat, mdp_trn, EPSILON, start_state)
                checkpoint[key] = {"mle": m, "itl": i}
                _atomic_save(checkpoint, ckpt_path)

            for k in mle_runs:
                mle_runs[k].append(m[k]); itl_runs[k].append(i[k])

        row = {"coverage": coverage}
        for k in mle_runs:
            mm, ms = summarize_runs(mle_runs[k])
            im, ist = summarize_runs(itl_runs[k])
            row[f"mle_{k}_mean"], row[f"mle_{k}_std"] = mm, ms
            row[f"itl_{k}_mean"], row[f"itl_{k}_std"] = im, ist
        rows.append(row)
        print(f"  cov={coverage:.1f}  "
              f"NV_t  MLE: {row['mle_normalized_value_transfer_mean']:+.3f}  "
              f"ITL: {row['itl_normalized_value_transfer_mean']:+.3f}    "
              f"BM_t  MLE: {row['mle_best_matching_transfer_mean']:.3f}  "
              f"ITL: {row['itl_best_matching_transfer_mean']:.3f}")

    out_path = f"results/tables/gridworld_transfer_{sf_tag}.json"
    os.makedirs("results/tables", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {"gamma": GAMMA, "delta": DELTA, "epsilon": EPSILON,
                       "stochastic_fraction": stochastic_fraction,
                       "n_seeds": n_seeds, "coverages": COVERAGES},
            "rows": rows,
        }, f, indent=2)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Randomworld transfer
# ---------------------------------------------------------------------------

def run_randomworld_transfer(stochastic_fraction, n_worlds, n_datasets):
    print("=" * 70)
    print(f"  RANDOMWORLD TRANSFER  (stochastic = {stochastic_fraction:.0%})")
    print("=" * 70)

    sf_tag = f"sf{int(round(stochastic_fraction * 100)):03d}"
    ckpt_path = f"results/checkpoints/randomworld_transfer_{sf_tag}.json"
    os.makedirs("results/checkpoints", exist_ok=True)
    checkpoint = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

    rows = []
    for coverage in COVERAGES:
        mle_runs = {k: [] for k in ["normalized_value_transfer", "best_matching_transfer",
                                     "epsilon_matching_transfer", "n_violations_transfer"]}
        itl_runs = {k: [] for k in mle_runs}

        for world_seed in range(n_worlds):
            mdp_std = make_randomworld(n_states=15, n_actions=5, gamma=GAMMA,
                                        n_successors=5, seed=world_seed)
            mdp_trn = make_randomworld_transfer(mdp_std, seed=world_seed + 1000)
            pi_expert = make_epsilon_optimal_expert(
                mdp_std, epsilon=EPSILON, target_stochastic_fraction=stochastic_fraction,
            )

            for data_seed in range(n_datasets):
                key = f"{coverage}:{world_seed}:{data_seed}"
                if key in checkpoint:
                    m = checkpoint[key]["mle"]; i = checkpoint[key]["itl"]
                else:
                    combined_seed = 1000 * world_seed + data_seed
                    N, T_mle, _ = generate_batch_dataset(
                        mdp_std, pi_expert, coverage=coverage, K=5,
                        delta=DELTA, seed=combined_seed,
                    )
                    T_hat, _ = solve_itl(N, T_mle, mdp_std.R, mdp_std.gamma,
                                          epsilon=EPSILON, max_iter=10, verbose=False)
                    m = _evaluate_transfer(T_mle, mdp_trn, EPSILON, start_state=None)
                    i = _evaluate_transfer(T_hat, mdp_trn, EPSILON, start_state=None)
                    checkpoint[key] = {"mle": m, "itl": i}
                    _atomic_save(checkpoint, ckpt_path)

                for k in mle_runs:
                    mle_runs[k].append(m[k]); itl_runs[k].append(i[k])

        row = {"coverage": coverage}
        for k in mle_runs:
            mm, ms = summarize_runs(mle_runs[k])
            im, ist = summarize_runs(itl_runs[k])
            row[f"mle_{k}_mean"], row[f"mle_{k}_std"] = mm, ms
            row[f"itl_{k}_mean"], row[f"itl_{k}_std"] = im, ist
        rows.append(row)
        print(f"  cov={coverage:.1f}  "
              f"NV_t  MLE: {row['mle_normalized_value_transfer_mean']:+.3f}  "
              f"ITL: {row['itl_normalized_value_transfer_mean']:+.3f}    "
              f"BM_t  MLE: {row['mle_best_matching_transfer_mean']:.3f}  "
              f"ITL: {row['itl_best_matching_transfer_mean']:.3f}")

    out_path = f"results/tables/randomworld_transfer_{sf_tag}.json"
    os.makedirs("results/tables", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {"gamma": GAMMA, "delta": DELTA, "epsilon": EPSILON,
                       "stochastic_fraction": stochastic_fraction,
                       "n_worlds": n_worlds, "n_datasets_per_world": n_datasets,
                       "coverages": COVERAGES},
            "rows": rows,
        }, f, indent=2)
    print(f"  → {out_path}")


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    env = os.environ.get("ENV", "gridworld")
    sf = float(os.environ.get("STOCHASTIC_FRACTION", "0.4"))

    if env == "gridworld":
        run_gridworld_transfer(sf, int(os.environ.get("N_SEEDS", "10")))
    elif env == "randomworld":
        run_randomworld_transfer(
            sf,
            int(os.environ.get("N_WORLDS", "5")),
            int(os.environ.get("N_DATASETS", "2")),
        )
    elif env == "both":
        run_gridworld_transfer(sf, int(os.environ.get("N_SEEDS", "10")))
        run_randomworld_transfer(
            sf,
            int(os.environ.get("N_WORLDS", "5")),
            int(os.environ.get("N_DATASETS", "2")),
        )
    else:
        raise ValueError(f"Unknown ENV={env}; pick gridworld / randomworld / both")


if __name__ == "__main__":
    main()
