"""
Expert policy simulation and trajectory data generation.

The expert is epsilon-optimal: it picks actions within the epsilon-ball
of the optimal Q-values. For deterministic experts, it always picks
the optimal action. For stochastic experts, it samples uniformly from
the epsilon-ball.
"""

import numpy as np
from typing import Tuple, Optional
from .mdp import TabularMDP


def generate_expert_trajectories(
    mdp: TabularMDP,
    pi_expert: np.ndarray,
    n_trajectories: int = 100,
    max_steps: int = 50,
    seed: Optional[int] = None,
) -> list:
    """
    Simulate expert trajectories in the MDP.

    Args:
        mdp: the environment
        pi_expert: expert policy, shape (n_states, n_actions)
        n_trajectories: how many episodes to simulate
        max_steps: max steps per episode
        seed: random seed

    Returns:
        trajectories: list of lists of (s, a, s') tuples
    """
    rng = np.random.default_rng(seed)
    trajectories = []

    for _ in range(n_trajectories):
        s = rng.integers(mdp.n_states)  # random start state
        traj = []
        for _ in range(max_steps):
            # Sample action from expert policy
            a = rng.choice(mdp.n_actions, p=pi_expert[s])
            # Sample next state from true dynamics
            s_next = rng.choice(mdp.n_states, p=mdp.T[s, a])
            traj.append((s, a, s_next))
            s = s_next
        trajectories.append(traj)

    return trajectories


def trajectories_to_counts(
    trajectories: list,
    n_states: int,
    n_actions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert trajectory data to transition counts and MLE estimates.

    This produces:
      N[s, a, s'] = number of times (s, a, s') appears in data
      T_mle[s, a, s'] = N[s, a, s'] / sum_{s''} N[s, a, s'']

    For state-action pairs never visited, T_mle is set to uniform (1/n_states).

    Args:
        trajectories: list of lists of (s, a, s') tuples
        n_states: number of states
        n_actions: number of actions

    Returns:
        N: count array, shape (n_states, n_actions, n_states)
        T_mle: MLE transition estimate, shape (n_states, n_actions, n_states)
    """
    N = np.zeros((n_states, n_actions, n_states))

    for traj in trajectories:
        for (s, a, s_next) in traj:
            N[s, a, s_next] += 1

    # MLE: normalize counts. For unvisited (s,a) pairs, use uniform.
    T_mle = np.zeros_like(N)
    for s in range(n_states):
        for a in range(n_actions):
            total = N[s, a].sum()
            if total > 0:
                T_mle[s, a] = N[s, a] / total
            else:
                T_mle[s, a] = 1.0 / n_states  # uniform prior for unvisited

    return N, T_mle


def make_stochastic_expert(
    mdp: TabularMDP,
    epsilon: float,
) -> np.ndarray:
    """
    Create a stochastic epsilon-optimal expert policy.

    The expert plays uniformly over actions within the epsilon-ball:
        pi(a|s) = 1/|E_eps(s)| if a in E_eps(s), else 0

    This matches the paper's assumption that the expert is near-optimal
    but not necessarily deterministic.

    Args:
        mdp: the MDP (uses optimal Q-values to determine epsilon-ball)
        epsilon: near-optimality tolerance

    Returns:
        pi_expert: shape (n_states, n_actions), stochastic policy
    """
    _, Q_star, _ = mdp.compute_optimal_policy()
    valid = mdp.compute_epsilon_ball(Q_star, epsilon)

    pi_expert = np.zeros((mdp.n_states, mdp.n_actions))
    for s in range(mdp.n_states):
        n_valid = valid[s].sum()
        if n_valid > 0:
            pi_expert[s, valid[s]] = 1.0 / n_valid
        else:
            pi_expert[s] = 1.0 / mdp.n_actions

    return pi_expert


def generate_batch_data(
    mdp: TabularMDP,
    pi_expert: np.ndarray,
    n_samples_per_sa: int = 10,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate batch data D = {(s, a, s')} by sampling n_samples_per_sa
    transitions for each (s, a) pair where the expert has nonzero policy.

    This matches the paper's experimental setup more directly than
    trajectory-based sampling.

    Returns:
        N: count array
        T_mle: MLE estimates
    """
    rng = np.random.default_rng(seed)
    N = np.zeros((mdp.n_states, mdp.n_actions, mdp.n_states))

    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            if pi_expert[s, a] > 0:
                # Expert visits this (s, a) pair
                for _ in range(n_samples_per_sa):
                    s_next = rng.choice(mdp.n_states, p=mdp.T[s, a])
                    N[s, a, s_next] += 1

    # Compute MLE
    T_mle = np.zeros_like(N)
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            total = N[s, a].sum()
            if total > 0:
                T_mle[s, a] = N[s, a] / total
            else:
                T_mle[s, a] = 1.0 / mdp.n_states

    return N, T_mle
