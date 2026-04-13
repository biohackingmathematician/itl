"""
Expert policy construction and batch-data generation, matching
Benac et al. (2024), Appendix "Generating batch data D":

  1. Select a fraction (coverage %) of states to be covered by D.
  2. For each selected state s, identify actions in epsilon(s; T*).
  3. Sample K transitions per (s, a) in the ε-ball.
  4. Dataset = {(s, a, s')} collected across those (s, a).

The ε-optimal expert is controlled by the number of stochastic-policy
states (|epsilon-ball(s)| > 1). The paper's main results use ~40%
stochastic-policy states; appendix includes 20% and 0%.

Laplace smoothing (Eq 5 of paper) is applied when computing T_MLE.

Textbook cross-reference (Krause & Hubotter, "Probabilistic Artificial
Intelligence"):
  - Ch 11 (Model-based Bayesian RL): each row T(.|s,a) is a categorical
    distribution over next states, so a Dir(delta, ..., delta) prior is
    conjugate and the posterior after counts N_{s,a} is Dir(N_{s,a} + delta).
    Laplace-smoothed MLE below is the posterior-mean estimator under this
    Dirichlet prior (or equivalently the MAP under Dir(1+delta)).
  - Definition 1 (eps-ball) and the eps-optimal expert use Q* from Ch 10
    Eq 10.9 / Def 10.7.
"""

import numpy as np
from typing import Tuple, Optional
from .mdp import TabularMDP


# =============================================================================
# MLE with Laplace smoothing (Eq 5 of the paper)
# =============================================================================

def compute_mle_transitions(
    N: np.ndarray,
    delta: float = 0.001,
) -> np.ndarray:
    """
    T_MLE(s'|s, a) = (N_{s,a,s'} + delta) / sum_{s''} (N_{s,a,s''} + delta).

    For unvisited (s, a) pairs, this reduces to uniform 1/n_states, which
    matches the paper's "uniform prior for unobserved (s, a)".

    This is the posterior-mean estimator of a Dirichlet-Categorical
    conjugate model with prior Dir(delta, ..., delta) — see Krause &
    Hubotter Ch 11 (Bayesian model-based RL, Dirichlet priors on
    categorical transitions).
    """
    smoothed = N + delta
    T_mle = smoothed / smoothed.sum(axis=2, keepdims=True)
    return T_mle


# =============================================================================
# Expert policy construction
# =============================================================================

def make_epsilon_optimal_expert(
    mdp: TabularMDP,
    epsilon: float,
    target_stochastic_fraction: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Construct an ε-optimal expert policy.

    The ε-ball at each state contains all actions with Q*-gap ≤ ε. When
    |ε-ball(s)| > 1, s is a *stochastic-policy state* (per Def 4 of the paper).

    If `target_stochastic_fraction` is provided, we adjust the per-state
    epsilon so that approximately that fraction of states are stochastic-
    policy states (by using a per-state epsilon that achieves the quantile
    of the next-best-action gap). This matches the paper's main setup
    (~40% stochastic-policy states).

    Returns pi_expert[s, a]: uniform over actions in the ε-ball at s.
    """
    _, Q_star, _ = mdp.compute_optimal_policy()
    n_states, n_actions = mdp.n_states, mdp.n_actions

    if target_stochastic_fraction is None:
        valid = mdp.compute_epsilon_ball(Q_star, epsilon)
    else:
        # Per-state ε-ball construction: for each state, compute the gap
        # between best and second-best action. Mark the `target_stochastic_fraction`
        # states with the SMALLEST such gap as stochastic-policy states (i.e.,
        # the states where the expert is most genuinely ambivalent).
        sorted_q = np.sort(Q_star, axis=1)[:, ::-1]  # descending
        best_to_second_gap = sorted_q[:, 0] - sorted_q[:, 1]

        n_stochastic = int(round(target_stochastic_fraction * n_states))
        # Argsort ascending; smallest gaps first
        order = np.argsort(best_to_second_gap)
        stochastic_states = set(order[:n_stochastic].tolist())

        valid = np.zeros((n_states, n_actions), dtype=bool)
        for s in range(n_states):
            if s in stochastic_states:
                # Include all actions whose Q* is within epsilon of the max.
                valid[s] = (Q_star[s].max() - Q_star[s]) <= epsilon + 1e-12
                # Make sure at least the best and second-best are in.
                top_two = np.argsort(Q_star[s])[::-1][:2]
                valid[s, top_two] = True
            else:
                # Deterministic-policy state: only the best action.
                best = Q_star[s].argmax()
                valid[s, best] = True

    pi_expert = np.zeros((n_states, n_actions))
    for s in range(n_states):
        n_valid = int(valid[s].sum())
        if n_valid > 0:
            pi_expert[s, valid[s]] = 1.0 / n_valid
        else:
            pi_expert[s, Q_star[s].argmax()] = 1.0

    return pi_expert


# =============================================================================
# Batch data generation (coverage-based, per paper appendix)
# =============================================================================

def generate_batch_dataset(
    mdp: TabularMDP,
    pi_expert: np.ndarray,
    coverage: float = 1.0,
    K: int = 10,
    delta: float = 0.001,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single batch dataset D per the paper's appendix procedure.

    Args:
        mdp: the true environment
        pi_expert: ε-optimal expert policy, shape (n_states, n_actions)
        coverage: fraction of states to include in D (0 < coverage ≤ 1)
        K: number of transitions per (s, a) in the ε-ball (paper: 10 GW, 5 RW)
        delta: Laplace smoothing constant for MLE (paper: 0.001)
        seed: RNG seed

    Returns:
        N: transition counts, shape (n_states, n_actions, n_states)
        T_mle: MLE estimates with Laplace smoothing, same shape
        covered_states: boolean array of shape (n_states,) — which states are in D
    """
    rng = np.random.default_rng(seed)
    n_states = mdp.n_states
    n_actions = mdp.n_actions

    # Step 1: Select coverage % of states
    n_covered = max(1, int(round(coverage * n_states)))
    covered = rng.choice(n_states, size=n_covered, replace=False)
    covered_mask = np.zeros(n_states, dtype=bool)
    covered_mask[covered] = True

    # Steps 2–3: For each covered state, sample K transitions for each action
    # the expert might take (i.e., each a with pi_expert[s, a] > 0).
    N = np.zeros((n_states, n_actions, n_states))
    for s in covered:
        for a in range(n_actions):
            if pi_expert[s, a] > 0:
                next_states = rng.choice(n_states, size=K, p=mdp.T[s, a])
                for s_next in next_states:
                    N[s, a, s_next] += 1

    T_mle = compute_mle_transitions(N, delta=delta)

    return N, T_mle, covered_mask


# =============================================================================
# Episode-based trajectory generation (kept for the corridor demo)
# =============================================================================

def generate_expert_trajectories(
    mdp: TabularMDP,
    pi_expert: np.ndarray,
    n_trajectories: int = 100,
    max_steps: int = 50,
    start_state: Optional[int] = None,
    seed: Optional[int] = None,
) -> list:
    """Simulate expert trajectories (kept as a convenience for demos)."""
    rng = np.random.default_rng(seed)
    trajectories = []
    for _ in range(n_trajectories):
        s = start_state if start_state is not None else rng.integers(mdp.n_states)
        traj = []
        for _ in range(max_steps):
            a = rng.choice(mdp.n_actions, p=pi_expert[s])
            s_next = rng.choice(mdp.n_states, p=mdp.T[s, a])
            traj.append((s, a, s_next))
            s = s_next
        trajectories.append(traj)
    return trajectories


def trajectories_to_counts(
    trajectories: list,
    n_states: int,
    n_actions: int,
    delta: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Count transitions from trajectories and compute smoothed MLE."""
    N = np.zeros((n_states, n_actions, n_states))
    for traj in trajectories:
        for (s, a, s_next) in traj:
            N[s, a, s_next] += 1
    T_mle = compute_mle_transitions(N, delta=delta)
    return N, T_mle


# =============================================================================
# Backwards-compatibility shims (for run_corridor.py, run_bitl.py, run_mimic.py)
# =============================================================================

def generate_batch_data(
    mdp: TabularMDP,
    pi_expert: np.ndarray,
    n_samples_per_sa: int = 10,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy wrapper: sample K per (s,a) over ALL states (coverage=1.0)."""
    N, T_mle, _ = generate_batch_dataset(
        mdp, pi_expert, coverage=1.0, K=n_samples_per_sa, seed=seed
    )
    return N, T_mle


def make_stochastic_expert(mdp: TabularMDP, epsilon: float) -> np.ndarray:
    """Legacy: uniform over the ε-ball using a single global epsilon."""
    return make_epsilon_optimal_expert(mdp, epsilon, target_stochastic_fraction=None)
