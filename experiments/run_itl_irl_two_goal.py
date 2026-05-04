"""
Stress test for ITL+IRL on the two-goal (non-goal-dominated) gridworld.

Motivation: the existing 5x5 gridworld is goal-dominated — see
`docs/c_itl_options.md` "Methodology gap discovered 2026-05-01" — so the
acceptance test "best-action match >= 0.7" can be passed by a solver
that only locates the goal, not one that recovers the reward
*structure*. The two-goal gridworld (`make_two_goal_gridworld`) breaks
that degeneracy: with goal A at top-right (R_A=5) and goal B at
bottom-left (R_B=10), separated by a thick-diagonal soft-wall barrier,
a "+10 at both goals" trivial reward gives a measurably worse policy
than the true reward.

This script runs five methods at coverage=1.0 with a 40% stochastic
expert across 20 seeds and reports per-method best_matching to pi*:

  1. MLE-T + R_TRUE       (just MLE dynamics, true reward)
  2. MLE-T + R_trivial    (just MLE dynamics, equal +10 at both goals)
  3. ITL-T + R_TRUE       (ITL with the correct reward as input)
  4. ITL-T + R_trivial    (ITL given the trivial reward as input)
  5. ITL+IRL              (joint T + w inference, R_hat = Phi @ w_hat)

ITL+IRL uses one-hot (s, a) features and a single anchor at goal B's
absorbing action with value 10.0 (matches the true reward at that
cell). The acceptance test for the methodology — implemented as a
unit test in `tests/test_smoke.py::test_itl_irl_recovers_R_on_two_goal`
— requires ITL+IRL to beat both trivial baselines by at least 0.05 mean
best-matching.

Outputs JSON to `results/tables/two_goal_itl_irl_sf040.json` and a
small console summary.
"""

import os
import sys
import json
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_two_goal_gridworld, two_goal_states
from src.expert import generate_batch_dataset, make_epsilon_optimal_expert
from src.itl_solver import solve_itl
from src.itl_irl_solver import solve_itl_irl
from src.mdp import TabularMDP
from src.utils import best_matching


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _onehot_sa_features(n_states: int, n_actions: int) -> np.ndarray:
    """One-hot features over (s, a) pairs; d = n_states * n_actions."""
    d = n_states * n_actions
    Phi = np.zeros((n_states, n_actions, d))
    for s in range(n_states):
        for a in range(n_actions):
            Phi[s, a, s * n_actions + a] = 1.0
    return Phi


def _r_trivial(n_states: int, n_actions: int, goals: Tuple[int, int],
               value: float = 10.0) -> np.ndarray:
    """+value at both goal cells (all actions), 0 elsewhere."""
    R = np.zeros((n_states, n_actions))
    for g in goals:
        R[g, :] = value
    return R


def _policy_match(T_hat: np.ndarray, R_used: np.ndarray, mdp,
                  pi_star: np.ndarray) -> float:
    """best_matching of pi*(T_hat, R_used) against pi*(T*, R*)."""
    mdp_hat = TabularMDP(mdp.n_states, mdp.n_actions, T_hat, R_used, mdp.gamma)
    _, _, pi_hat = mdp_hat.compute_optimal_policy()
    return float(best_matching(pi_hat, pi_star))


def _summarize(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


# ----------------------------------------------------------------------
# Per-seed run — re-uses checkpoint entries when present
# ----------------------------------------------------------------------


def run_one_seed(mdp, pi_expert, goals: Tuple[int, int], R_trivial: np.ndarray,
                 Phi: np.ndarray, anchor: Tuple[int, int, float],
                 epsilon: float, K: int, delta: float, lambda_l1: float,
                 seed: int, max_iter: int) -> Dict[str, float]:
    """Compute best_matching for all five methods on a single seed."""
    N, T_mle, _ = generate_batch_dataset(
        mdp, pi_expert, coverage=1.0, K=K, delta=delta, seed=seed,
    )
    _, _, pi_star = mdp.compute_optimal_policy()

    out: Dict[str, float] = {}

    # 1. MLE-T + R_TRUE
    out["mle_true"] = _policy_match(T_mle, mdp.R, mdp, pi_star)

    # 2. MLE-T + R_trivial
    out["mle_triv"] = _policy_match(T_mle, R_trivial, mdp, pi_star)

    # 3. ITL-T + R_TRUE  (ITL given the true R; eval under true R)
    T_itl_true, _ = solve_itl(
        N, T_mle, mdp.R, mdp.gamma, epsilon=epsilon, max_iter=10, verbose=False,
    )
    out["itl_true"] = _policy_match(T_itl_true, mdp.R, mdp, pi_star)

    # 4. ITL-T + R_trivial  (ITL given the trivial R; eval under that R)
    T_itl_triv, _ = solve_itl(
        N, T_mle, R_trivial, mdp.gamma, epsilon=epsilon, max_iter=10,
        verbose=False,
    )
    out["itl_triv"] = _policy_match(T_itl_triv, R_trivial, mdp, pi_star)

    # 5. ITL+IRL  (joint inference; eval under R_hat = Phi @ w_hat)
    T_irl, w_irl, info = solve_itl_irl(
        N, T_mle, Phi, mdp.gamma, epsilon=epsilon,
        anchor=anchor, lambda_l1=lambda_l1, max_iter=max_iter, tol=1e-6,
        verbose=False,
    )
    R_irl = Phi @ w_irl
    out["itl_irl"] = _policy_match(T_irl, R_irl, mdp, pi_star)
    out["_itl_irl_iters"] = float(len(info.get("iterations", [])))
    out["_itl_irl_termination"] = info.get("termination", "")
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("  ITL+IRL stress test on the two-goal gridworld (non-goal-dominated)")
    print("=" * 72)

    # Paper-aligned constants for Gridworld (40% stochastic expert).
    GAMMA = 0.95
    DELTA = 0.001
    EPSILON = 5.0
    K = 10
    SLIP = 0.2
    STOCHASTIC_FRACTION = 0.4
    # lambda_l1 = 0.1 reaches the same fixed-point in 2 outer iterations as
    # 0.01 needs 10 (each call ~20s vs ~150s). The recovered w is identical
    # at the anchor (10) with the rest pushed to ~0 by the L1 prior; the
    # resulting best_matching is the same.
    LAMBDA_L1 = 0.1
    GRID_SIZE = 5

    # Spec: 20 seeds, not 50, to keep runtime modest.
    N_SEEDS = int(os.environ.get("N_SEEDS", "20"))
    MAX_OUTER_IRL = int(os.environ.get("ITL_IRL_MAX_OUTER", "10"))

    mdp = make_two_goal_gridworld(
        grid_size=GRID_SIZE, gamma=GAMMA, slip_prob=SLIP,
    )
    n_states, n_actions = mdp.n_states, mdp.n_actions
    goal_A, goal_B = two_goal_states(grid_size=GRID_SIZE)
    print(f"  States: {n_states}  Actions: {n_actions}  gamma: {mdp.gamma}")
    print(f"  Goal A (R={mdp.R[goal_A, 0]:.1f}): state {goal_A} (top-right)")
    print(f"  Goal B (R={mdp.R[goal_B, 0]:.1f}): state {goal_B} (bottom-left)")

    pi_expert = make_epsilon_optimal_expert(
        mdp, epsilon=EPSILON, target_stochastic_fraction=STOCHASTIC_FRACTION,
    )
    n_stoch = int(((pi_expert > 0).sum(axis=1) > 1).sum())
    print(f"  Expert: {n_stoch}/{n_states} stochastic-policy states "
          f"(target {STOCHASTIC_FRACTION:.0%})")

    R_trivial = _r_trivial(n_states, n_actions, (goal_A, goal_B), value=10.0)
    Phi = _onehot_sa_features(n_states, n_actions)
    anchor = (goal_B, 0, 10.0)
    print(f"  Phi shape: {Phi.shape}  anchor: R(s={anchor[0]}, a={anchor[1]}) "
          f"= {anchor[2]}  lambda_l1: {LAMBDA_L1}")

    # ------------------------------------------------------------------
    # Per-seed checkpointed sweep
    # ------------------------------------------------------------------
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    ckpt_path = "results/checkpoints/two_goal_itl_irl_sf040.json"
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
    else:
        ckpt = {}

    methods = ["mle_true", "mle_triv", "itl_true", "itl_triv", "itl_irl"]
    per_method: Dict[str, List[float]] = {m: [] for m in methods}

    print(f"\n  Running {N_SEEDS} seeds ...")
    for seed in range(N_SEEDS):
        key = str(seed)
        entry = ckpt.get(key)
        if entry is None or any(m not in entry for m in methods):
            metrics = run_one_seed(
                mdp, pi_expert, (goal_A, goal_B), R_trivial,
                Phi, anchor, EPSILON, K, DELTA, LAMBDA_L1,
                seed=seed, max_iter=MAX_OUTER_IRL,
            )
            ckpt[key] = metrics
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f)
        else:
            metrics = entry

        for m in methods:
            per_method[m].append(float(metrics[m]))

        # Per-seed log line
        bits = "  ".join(f"{m}={metrics[m]:.3f}" for m in methods)
        iters = int(metrics.get("_itl_irl_iters", 0))
        print(f"    seed={seed:>2d}  {bits}   "
              f"(itl+irl iters={iters})")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    summary = {}
    for m in methods:
        mean_, std_ = _summarize(per_method[m])
        summary[m] = {"mean": mean_, "std": std_}

    out_path = "results/tables/two_goal_itl_irl_sf040.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "grid_size": GRID_SIZE, "gamma": GAMMA, "slip": SLIP,
                "epsilon": EPSILON, "K": K, "delta": DELTA,
                "stochastic_fraction": STOCHASTIC_FRACTION,
                "lambda_l1": LAMBDA_L1, "n_seeds": N_SEEDS,
                "anchor": list(anchor),
                "goal_A": goal_A, "goal_B": goal_B,
            },
            "summary": summary,
            "per_method": per_method,
        }, f, indent=2)

    print("\n" + "-" * 72)
    print(f"  best_matching across {N_SEEDS} seeds (mean ± std)")
    print("-" * 72)
    for m in methods:
        s = summary[m]
        print(f"    {m:<12s} {s['mean']:.3f} ± {s['std']:.3f}")

    # Acceptance: ITL+IRL must beat both trivial baselines by >= 0.05.
    irl = summary["itl_irl"]["mean"]
    mle_t = summary["mle_triv"]["mean"]
    itl_t = summary["itl_triv"]["mean"]
    gap_vs_mle_triv = irl - mle_t
    gap_vs_itl_triv = irl - itl_t

    print("\n  Acceptance criteria (each must be >= 0.05):")
    print(f"    ITL+IRL - (MLE-T + R_trivial) = {gap_vs_mle_triv:+.3f}  "
          f"{'PASS' if gap_vs_mle_triv >= 0.05 else 'FAIL'}")
    print(f"    ITL+IRL - (ITL-T + R_trivial) = {gap_vs_itl_triv:+.3f}  "
          f"{'PASS' if gap_vs_itl_triv >= 0.05 else 'FAIL'}")

    accept = gap_vs_mle_triv >= 0.05 and gap_vs_itl_triv >= 0.05
    print(f"\n  OVERALL ACCEPTANCE: {'PASS' if accept else 'FAIL'}")
    print(f"  Output: {out_path}")
    return 0 if accept else 1


if __name__ == "__main__":
    sys.exit(main())
