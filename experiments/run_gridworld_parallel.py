"""
Round-robin seed runner for the gridworld coverage sweep.

Reuses the same checkpoint file (results/checkpoints/gridworld_seeds.json) as
run_gridworld.py, but iterates seeds in the OUTER loop instead of coverage.
Each chunk of wall-clock time therefore advances every coverage evenly, which
makes resumption-after-timeout a lot more useful when running under a strict
per-call budget.

Run repeatedly with the same N_SEEDS to fill the gap. Re-runs are no-ops once
all (coverage, seed) pairs are present.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_gridworld, gridworld_start_state
from src.expert import make_epsilon_optimal_expert, generate_batch_dataset
from src.itl_solver import solve_itl
from src.utils import (
    best_matching, epsilon_matching, normalized_value, total_variation,
    count_constraint_violations, transition_mse,
)
from src.mdp import TabularMDP


GAMMA, DELTA, EPSILON, STOCHASTIC_FRACTION, K = 0.95, 0.001, 5.0, 0.4, 10
COVERAGES = [0.2, 0.4, 0.6, 0.8, 1.0]


def evaluate(T_hat, R, gamma, true_mdp, epsilon, start_state):
    mdp_l = TabularMDP(true_mdp.n_states, true_mdp.n_actions, T_hat, R, gamma)
    _, _, pi_hat = mdp_l.compute_optimal_policy()
    _, Q_star, pi_star = true_mdp.compute_optimal_policy()
    return {
        "normalized_value": normalized_value(T_hat, R, gamma, true_mdp, start_state=start_state),
        "best_matching": best_matching(pi_hat, pi_star),
        "epsilon_matching": epsilon_matching(pi_hat, Q_star, epsilon),
        "total_variation": total_variation(true_mdp.T, T_hat),
        "n_violations": count_constraint_violations(T_hat, R, gamma, true_mdp, epsilon),
        "mse": transition_mse(true_mdp.T, T_hat),
    }


def main():
    n_seeds = int(os.environ.get("N_SEEDS", "20"))
    deadline = time.time() + float(os.environ.get("BUDGET_S", "35"))

    mdp = make_gridworld(grid_size=5, gamma=GAMMA, slip_prob=0.2, transfer=False)
    pi_expert = make_epsilon_optimal_expert(
        mdp, epsilon=EPSILON, target_stochastic_fraction=STOCHASTIC_FRACTION,
    )
    start_state = gridworld_start_state(grid_size=5)

    ckpt_path = "results/checkpoints/gridworld_seeds.json"
    os.makedirs("results/checkpoints", exist_ok=True)
    checkpoint = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

    completed_initial = len(checkpoint)
    new_pairs = 0

    # OUTER seed, INNER coverage  -> each pass through the seeds advances all
    # coverages equally.
    for seed in range(n_seeds):
        for coverage in COVERAGES:
            key = f"{coverage}:{seed}"
            if key in checkpoint:
                continue
            if time.time() > deadline:
                _save(checkpoint, ckpt_path)
                _print_summary(checkpoint, completed_initial, new_pairs, "deadline")
                return
            N, T_mle, _ = generate_batch_dataset(
                mdp, pi_expert, coverage=coverage, K=K, delta=DELTA, seed=seed,
            )
            T_hat, _ = solve_itl(N, T_mle, mdp.R, mdp.gamma,
                                  epsilon=EPSILON, max_iter=10, verbose=False)
            checkpoint[key] = {
                "mle": evaluate(T_mle, mdp.R, mdp.gamma, mdp, EPSILON, start_state),
                "itl": evaluate(T_hat, mdp.R, mdp.gamma, mdp, EPSILON, start_state),
            }
            new_pairs += 1
            _save(checkpoint, ckpt_path)

    _print_summary(checkpoint, completed_initial, new_pairs, "all_done")


def _save(checkpoint, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(checkpoint, f)
    os.replace(tmp, path)


def _print_summary(checkpoint, before, new_pairs, why):
    from collections import Counter
    cov_count = Counter(k.split(":")[0] for k in checkpoint)
    print(f"  stop={why}  before={before} after={len(checkpoint)} (+{new_pairs})")
    for c in sorted(cov_count, key=float):
        print(f"    coverage={c}: {cov_count[c]} seeds")


if __name__ == "__main__":
    main()
