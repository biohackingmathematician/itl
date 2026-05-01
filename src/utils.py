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

Textbook cross-reference (Krause & Hubotter, "Probabilistic Artificial
Intelligence"):
  - Normalized Value and best/eps-matching are defined over V^pi and Q* from
    Ch 10 (Bellman evaluation Eq 10.20, Bellman optimality Eq 10.9 / Def 10.7).
  - Bayesian Regret is introduced in Ch 11 as the core metric for
    model-based Bayesian RL under posterior uncertainty over dynamics.
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

    Both V^pi and V^* are the Bellman-evaluation values from Krause & Hubotter
    Ch 10 (Eq 10.20 for evaluation under fixed policy; Eq 10.9 for optimality).
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
# Value CVaR (paper Table 4 metric)
# =============================================================================

def value_cvar_from_T_distribution(
    T_samples: np.ndarray,
    R: np.ndarray,
    gamma: float,
    true_mdp: TabularMDP,
    alpha: float = 0.05,
    start_state: Optional[int] = None,
) -> float:
    """
    Conditional Value-at-Risk of the policy value at level alpha.

    For each posterior / bootstrap sample T_i:
      pi_i  = pi*(T_i, R)             optimal policy under sampled dynamics
      V_i   = V^{pi_i}(s0; T*, R)     evaluated under TRUE dynamics

    CVaR_alpha = mean of V_i in the worst alpha-tail
              = mean(V_i | V_i <= percentile(V, alpha * 100))

    Paper Table 4 reports CVaR at alpha=0.01, 0.02, 0.05.

    Caveat: paper methodology section was not directly read this session.
    This implementation chooses the most natural definition (lower-tail
    average over the dynamics distribution, evaluated under the true T*).
    Verify against paper Section 5 before publication.

    Args:
        T_samples: shape (K, n_states, n_actions, n_states) — posterior or
            bootstrap samples of the dynamics.
        R: reward function (n_states, n_actions).
        gamma: discount factor.
        true_mdp: the TRUE MDP (its T is used for evaluation).
        alpha: tail level (e.g., 0.05 for "CVaR 5%").
        start_state: state to evaluate at; None averages over a uniform
            initial distribution.

    Returns:
        CVaR_alpha as a float.
    """
    K = T_samples.shape[0]
    n_states = true_mdp.n_states
    n_actions = true_mdp.n_actions

    V_samples = np.zeros(K)
    for i in range(K):
        mdp_i = TabularMDP(n_states, n_actions, T_samples[i], R, gamma)
        _, _, pi_i = mdp_i.compute_optimal_policy()
        V_under_true = true_mdp.compute_value_function(pi_i)
        V_samples[i] = (
            V_under_true[start_state] if start_state is not None
            else V_under_true.mean()
        )

    # Lower tail mean
    threshold = np.percentile(V_samples, alpha * 100.0)
    tail = V_samples[V_samples <= threshold]
    if len(tail) == 0:
        return float(V_samples.min())
    return float(tail.mean())


def bootstrap_T_samples(
    N: np.ndarray,
    delta: float = 0.001,
    n_samples: int = 200,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample K transition matrices from Dir(N + delta) per (s, a). Used to give
    point-estimate methods (MLE, ITL, MCE) a distribution over T for CVaR.

    BITL doesn't need this — it has its own posterior. The paper's CVaR
    columns for non-Bayesian methods presumably use a similar bootstrap;
    this implementation is the natural choice but should be checked against
    paper methodology.
    """
    rng = np.random.default_rng(seed)
    n_states, n_actions, _ = N.shape
    samples = np.zeros((n_samples, n_states, n_actions, n_states))
    alpha = N + delta
    for s in range(n_states):
        for a in range(n_actions):
            # numpy Dirichlet expects 1-D concentration vector
            samples[:, s, a, :] = rng.dirichlet(alpha[s, a], size=n_samples)
    return samples


def value_cvar_from_point_T(
    T_hat: np.ndarray,
    N: np.ndarray,
    R: np.ndarray,
    gamma: float,
    true_mdp: TabularMDP,
    alpha: float = 0.05,
    start_state: Optional[int] = None,
    n_bootstrap: int = 200,
    delta: float = 0.001,
    seed: Optional[int] = None,
) -> float:
    """
    CVaR for a point-estimate method, "robustness of method's policy" variant.

    The method's policy is FIXED at pi* = pi*(T_hat, R). For each Dir(N+δ)
    bootstrap sample T_i, we evaluate that policy under T_i:
        V_i = V^{pi*}(s_0; T_i, R)
    CVaR_alpha is the lower-tail mean of {V_i}.

    This differs from `value_cvar_from_T_distribution` (used for BITL/PS),
    which re-derives a *new* optimal policy per posterior sample. The
    difference matters: for point-estimate methods we want "is the method's
    chosen policy robust to noise in T?", not "if T were T_i, what would
    the optimal policy do?"

    Pre-fix (≤ 2026-04-30) this function used the BITL-style version, with
    the consequence that MLE / ITL / MCE all produced *identical* CVaR
    numbers because the policy step ignored `T_hat` entirely. That meant
    the CVaR column couldn't distinguish methods.

    Caveat: paper methodology section was not directly read this session,
    so this is the most natural defensible interpretation rather than a
    verified replica. Treat the CVaR column as "qualitative differentiation
    of methods" until verified.

    Args:
        T_hat: the method's point estimate of T.
        N: transition counts, used only to define the bootstrap distribution.
        R: reward function (n_states, n_actions).
        gamma: discount factor.
        true_mdp: true MDP whose dynamics define the bootstrap base.
        alpha: tail level (0.01, 0.02, 0.05 in the paper).
        start_state: state to evaluate at; None averages over uniform initial.
        n_bootstrap: number of bootstrap samples K.
        delta: Dirichlet pseudo-count.
        seed: RNG seed.
    """
    n_states = true_mdp.n_states
    n_actions = true_mdp.n_actions

    # Fix the method's policy.
    mdp_method = TabularMDP(n_states, n_actions, T_hat, R, gamma)
    _, _, pi_method = mdp_method.compute_optimal_policy()

    # Bootstrap dynamics samples from Dir(N + delta).
    T_samples = bootstrap_T_samples(N, delta=delta, n_samples=n_bootstrap, seed=seed)

    V_samples = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        mdp_i = TabularMDP(n_states, n_actions, T_samples[i], R, gamma)
        V_under_i = mdp_i.compute_value_function(pi_method)
        V_samples[i] = (
            V_under_i[start_state] if start_state is not None
            else V_under_i.mean()
        )

    threshold = np.percentile(V_samples, alpha * 100.0)
    tail = V_samples[V_samples <= threshold]
    if len(tail) == 0:
        return float(V_samples.min())
    return float(tail.mean())


# =============================================================================
# Aggregation helpers
# =============================================================================

def summarize_runs(values: list) -> Tuple[float, float]:
    """Return (mean, std) over a list of scalar values.

    Edge cases:
      - empty list → (NaN, NaN), no warning
      - single element → (value, 0.0)
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


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
