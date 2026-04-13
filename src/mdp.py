"""
MDP definition, value iteration, and Q-value computation.

Notation follows Benac et al. (2024) / Krause & Hubotter Ch10:
  S = set of states (indexed 0..n_states-1)
  A = set of actions (indexed 0..n_actions-1)
  T[s, a, s'] = P(s' | s, a) — transition probability
  R[s, a] = immediate reward
  gamma = discount factor
"""

import numpy as np
from typing import Optional


class TabularMDP:
    """A finite MDP with tabular transition and reward matrices."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        T: np.ndarray,        # shape (n_states, n_actions, n_states)
        R: np.ndarray,         # shape (n_states, n_actions)
        gamma: float = 0.95,   # paper default (Benac et al. 2024)
    ):
        assert T.shape == (n_states, n_actions, n_states)
        assert R.shape == (n_states, n_actions)
        assert 0 <= gamma < 1

        # Verify T is a valid transition matrix (rows sum to 1)
        row_sums = T.sum(axis=2)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"Transition rows must sum to 1. Max deviation: {np.max(np.abs(row_sums - 1))}"
        )

        self.n_states = n_states
        self.n_actions = n_actions
        self.T = T.copy()
        self.R = R.copy()
        self.gamma = gamma

    def compute_policy_matrices(self, pi: np.ndarray):
        """
        Given policy pi[s, a] = P(a | s), compute:
          P_pi[s, s'] = sum_a pi[s,a] * T[s,a,s']   (transition matrix under pi)
          r_pi[s]     = sum_a pi[s,a] * R[s,a]       (reward vector under pi)

        These are P^pi and r^pi from Ch10 Eq 10.18.
        """
        n_s, n_a = self.n_states, self.n_actions

        # P_pi[s, s'] = sum over a of pi(a|s) * T(s'|s,a)
        P_pi = np.einsum("sa,sab->sb", pi, self.T)

        # r_pi[s] = sum over a of pi(a|s) * R(s,a)
        r_pi = np.einsum("sa,sa->s", pi, self.R)

        return P_pi, r_pi

    def compute_value_function(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute v^pi = (I - gamma * P^pi)^{-1} r^pi
        This is Ch10 Eq 10.20, the central equation ITL uses as a constraint.
        """
        P_pi, r_pi = self.compute_policy_matrices(pi)
        A = np.eye(self.n_states) - self.gamma * P_pi
        v = np.linalg.solve(A, r_pi)
        return v

    def compute_q_values(self, v: np.ndarray) -> np.ndarray:
        """
        Compute Q(s, a) = R(s, a) + gamma * sum_{s'} T(s, a, s') * v(s')
        This is Ch10 Eq 10.9 / Def 10.7.

        Returns Q of shape (n_states, n_actions).
        """
        # T[s, a, s'] @ v[s'] -> expected next-state value for each (s, a)
        expected_next = np.einsum("sab,b->sa", self.T, v)
        Q = self.R + self.gamma * expected_next
        return Q

    def compute_optimal_policy(self, tol: float = 1e-10, max_iter: int = 1000):
        """
        Value iteration to find optimal policy and value function.

        Returns:
            v_star: optimal value function, shape (n_states,)
            q_star: optimal Q-values, shape (n_states, n_actions)
            pi_star: deterministic optimal policy as (n_states, n_actions) one-hot
        """
        v = np.zeros(self.n_states)

        for _ in range(max_iter):
            Q = self.compute_q_values(v)
            v_new = Q.max(axis=1)
            if np.max(np.abs(v_new - v)) < tol:
                v = v_new
                break
            v = v_new

        Q = self.compute_q_values(v)
        pi = np.zeros((self.n_states, self.n_actions))
        pi[np.arange(self.n_states), Q.argmax(axis=1)] = 1.0

        return v, Q, pi

    def compute_epsilon_ball(self, Q: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Compute the epsilon-ball E_epsilon(s; T) for each state.

        Definition 1 from the paper:
          a in E_epsilon(s; T) iff max_{a'} Q*(s,a') - Q*(s,a) <= epsilon

        Returns:
            valid: boolean array of shape (n_states, n_actions)
                   valid[s, a] = True if action a is in the epsilon-ball at state s
        """
        q_max = Q.max(axis=1, keepdims=True)  # (n_states, 1)
        gaps = q_max - Q                       # (n_states, n_actions)
        valid = gaps <= epsilon + 1e-12        # small tolerance for numerics
        return valid


def deterministic_policy(n_states: int, n_actions: int, actions: np.ndarray) -> np.ndarray:
    """
    Create a deterministic policy matrix from an array of action indices.

    Args:
        n_states: number of states
        n_actions: number of actions
        actions: array of shape (n_states,) with action index per state

    Returns:
        pi: array of shape (n_states, n_actions) with one-hot rows
    """
    pi = np.zeros((n_states, n_actions))
    pi[np.arange(n_states), actions] = 1.0
    return pi
