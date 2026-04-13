"""
Metrics, plotting, and utilities for ITL experiments.

Paper metrics (Benac et al. 2024, Table 4):

  - Best matching: fraction of states where arg max pi(learned) == arg max pi*(T*)
  - ε-matching: fraction of states where arg max pi(learned) is in ε-ball(s; T*)
  - Normalized Value: V^{pi(learned)}(s_0; T*) / V^{pi*(T*)}(s_0; T*)
  - Bayesian Regret: (requires BITL posterior — not in MVR)
  - Total Variation: sum over (s, a) of 0.5 * ||T_hat(.|s,a) - T*(.|s,a)||_1
  - # constraints violated: count of ε-ball constraint violations on learned T

Dynamics metrics (for reference, not in paper):

  - MSE on transitions (used in archive_v1; kept for backward compat)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from .mdp import TabularMDP


# =============================================================================
# Paper metrics
# =============================================================================

def best_matching(pi_learned: np.ndarray, pi_star: np.ndarray) -> float:
    """Fraction of states where the greedy action under pi_learned matches pi_star."""
    a_learned = pi_learned.argmax(axis=1)
    a_star = pi_star.argmax(axis=1)
    return float(np.mean(a_learned == a_star))


def epsilon_matching(
    pi_learned: np.ndarray,
    Q_star: np.ndarray,
    epsilon: float,
) -> float:
    """Fraction of states where pi_learned's greedy action falls in ε-ball(s; T*)."""
    a_learned = pi_learned.argmax(axis=1)
    gaps = Q_star.max(axis=1) - Q_star[np.arange(Q_star.shape[0]), a_learned]
    return float(np.mean(gaps <= epsilon + 1e-12))


def normalized_value(
    T_hat: np.ndarray,
    R: np.ndarray,
    gamma: float,
    true_mdp: TabularMDP,
    start_state: Optional[int] = None,
) -> float:
    """
    Normalized Value:
        V^{pi*(T_hat)}(s0; T*) / V^{pi*(T*)}(s0; T*)

    If start_state is None, average over uniform initial distribution.
    Uses the reward function R = true_mdp.R (same reward on both sides).
    """
    n_states = true_mdp.n_states
    n_actions = true_mdp.n_actions

    # Optimal policy under learned dynamics
    mdp_learned = TabularMDP(n_states, n_actions, T_hat, R, gamma)
    _, _, pi_hat = mdp_learned.compute_optimal_policy()

    # Evaluate that policy under TRUE dynamics
    V_hat_under_true = true_mdp.compute_value_function(pi_hat)

    # True optimal value under true dynamics
    V_star, _, _ = true_mdp.compute_optimal_policy()

    if start_state is not None:
        num = V_hat_under_true[start_state]
        den = V_star[start_state]
    else:
        num = V_hat_under_true.mean()
        den = V_star.mean()

    if abs(den) < 1e-12:
        return 0.0
    return float(num / den)


def total_variation(T_true: np.ndarray, T_hat: np.ndarray) -> float:
    """Sum over (s, a) of 0.5 * ||T_hat(.|s,a) - T*(.|s,a)||_1."""
    return float(0.5 * np.sum(np.abs(T_hat - T_true)))


def count_constraint_violations(
    T_hat: np.ndarray,
    R: np.ndarray,
    gamma: float,
    true_mdp: TabularMDP,
    epsilon: float,
) -> int:
    """
    Count ε-ball property violations on T_hat: for each state s, if the optimal
    action under T_hat is NOT in ε-ball(s; T*), that's a violation.
    """
    n_states = true_mdp.n_states
    n_actions = true_mdp.n_actions

    _, Q_star, _ = true_mdp.compute_optimal_policy()
    valid_true = true_mdp.compute_epsilon_ball(Q_star, epsilon)

    mdp_learned = TabularMDP(n_states, n_actions, T_hat, R, gamma)
    _, _, pi_hat = mdp_learned.compute_optimal_policy()
    a_learned = pi_hat.argmax(axis=1)

    violations = 0
    for s in range(n_states):
        if not valid_true[s, a_learned[s]]:
            violations += 1
    return violations


# =============================================================================
# Dynamics MSE (kept for reference / regression tests)
# =============================================================================

def transition_mse(T_true: np.ndarray, T_hat: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        errors = []
        for s in range(T_true.shape[0]):
            for a in range(T_true.shape[1]):
                if mask[s, a]:
                    errors.append(np.sum((T_true[s, a] - T_hat[s, a]) ** 2))
        if len(errors) == 0:
            return 0.0
        return float(np.mean(errors))
    return float(np.mean(np.sum((T_true - T_hat) ** 2, axis=2)))


def transition_mse_visited_vs_unvisited(
    T_true: np.ndarray,
    T_hat: np.ndarray,
    N: np.ndarray,
) -> dict:
    """MSE split by whether the (s, a) pair was visited in the dataset."""
    sa_visited = N.sum(axis=2) > 0
    return {
        "mse_all": transition_mse(T_true, T_hat),
        "mse_visited": transition_mse(T_true, T_hat, mask=sa_visited),
        "mse_unvisited": transition_mse(T_true, T_hat, mask=~sa_visited),
        "n_visited": int(sa_visited.sum()),
        "n_unvisited": int((~sa_visited).sum()),
    }


# =============================================================================
# Aggregation helpers
# =============================================================================

def summarize_runs(values: list) -> Tuple[float, float]:
    """Return (mean, std) over a list of scalar values."""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


# =============================================================================
# Plotting
# =============================================================================

def plot_coverage_sensitivity(
    coverages: list,
    itl_mean: list,
    itl_std: list,
    mle_mean: list,
    mle_std: list,
    metric_name: str = "Normalized Value",
    title: str = "ITL vs MLE: Coverage Sensitivity",
    save_path: Optional[str] = None,
):
    """Plot metric vs coverage, with shaded standard deviation bands."""
    coverages = np.asarray(coverages)
    itl_mean = np.asarray(itl_mean); itl_std = np.asarray(itl_std)
    mle_mean = np.asarray(mle_mean); mle_std = np.asarray(mle_std)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(coverages, itl_mean, "o-", label="ITL", color="#2563eb", linewidth=2)
    ax.fill_between(coverages, itl_mean - itl_std, itl_mean + itl_std,
                    color="#2563eb", alpha=0.2)

    ax.plot(coverages, mle_mean, "s--", label="MLE", color="#dc2626", linewidth=2)
    ax.fill_between(coverages, mle_mean - mle_std, mle_mean + mle_std,
                    color="#dc2626", alpha=0.2)

    ax.set_xlabel("Coverage (fraction of states in D)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_results_table(results: dict, method_name: str = "ITL"):
    """Print a compact results table."""
    print(f"\n  {method_name}:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"    {k:<22s} {v:.4f}")
        else:
            print(f"    {k:<22s} {v}")
