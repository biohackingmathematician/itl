"""
Environment definitions for ITL reproduction.

Implements:
  1. Corridor (3-state) — toy example from Agna's notes for verification
  2. Gridworld (5x5, 25 states) — main synthetic benchmark from paper
  3. RandomWorld (15 states, 5 actions) — random MDP benchmark from paper
"""

import numpy as np
from typing import Optional
from .mdp import TabularMDP


# =============================================================================
# 1. CORRIDOR (3-state toy MDP from notes)
# =============================================================================

def make_corridor(gamma: float = 0.9) -> TabularMDP:
    """
    3-state corridor MDP from Agna's March 2026 worked example.

    States: s1, s2, s_goal (indices 0, 1, 2)
    Actions: Left=0, Right=1
    Rewards: R(s1, .) = R(s2, .) = -0.1, R(s_goal, .) = +10
    Dynamics: Right moves forward (80% success, 20% slip stay),
              Left moves backward (80% success, 20% slip).

    Known hand-computed values (gamma=0.9, epsilon=1.0):
      v* = [76.87, 87.68, 100.00]
      Q*(s1, R) = 76.87, Q*(s1, L) = 69.08
      Q*(s2, R) = 87.68, Q*(s2, L) = 71.03
    """
    n_states, n_actions = 3, 2  # L=0, R=1

    T = np.zeros((n_states, n_actions, n_states))

    # s1 (index 0)
    T[0, 1] = [0.2, 0.8, 0.0]   # s1, Right -> mostly s2
    T[0, 0] = [1.0, 0.0, 0.0]   # s1, Left -> stays at s1

    # s2 (index 1)
    T[1, 1] = [0.0, 0.2, 0.8]   # s2, Right -> mostly s_goal
    T[1, 0] = [0.8, 0.2, 0.0]   # s2, Left -> mostly back to s1

    # s_goal (index 2) — absorbing
    T[2, 0] = [0.0, 0.0, 1.0]
    T[2, 1] = [0.0, 0.0, 1.0]

    R = np.array([
        [-0.1, -0.1],   # s1
        [-0.1, -0.1],   # s2
        [10.0, 10.0],   # s_goal
    ])

    return TabularMDP(n_states, n_actions, T, R, gamma)


# =============================================================================
# 2. GRIDWORLD (5x5, 25 states, 4 actions)
# =============================================================================

def make_gridworld(
    grid_size: int = 5,
    gamma: float = 0.9,
    goal_reward: float = 10.0,
    step_cost: float = -0.1,
    slip_prob: float = 0.1,
    seed: Optional[int] = None,
) -> TabularMDP:
    """
    5x5 Gridworld matching the paper's synthetic benchmark.

    States: 25 cells in a grid (index = row * grid_size + col)
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    Dynamics: Move in intended direction with prob (1 - slip_prob),
              slip to random adjacent cell with prob slip_prob.
              Hitting a wall = stay in place.
    Reward: step_cost everywhere, goal_reward at bottom-right corner.

    Args:
        grid_size: side length of the grid
        gamma: discount factor
        goal_reward: reward at goal state
        step_cost: cost per step (negative)
        slip_prob: probability of slipping to adjacent cell
        seed: random seed (for any stochastic elements)
    """
    n_states = grid_size * grid_size
    n_actions = 4  # Up=0, Down=1, Left=2, Right=3

    def state_idx(r, c):
        return r * grid_size + c

    def state_rc(s):
        return s // grid_size, s % grid_size

    # Action effects: (delta_row, delta_col)
    action_deltas = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }

    goal_state = state_idx(grid_size - 1, grid_size - 1)

    T = np.zeros((n_states, n_actions, n_states))
    R = np.full((n_states, n_actions), step_cost)
    R[goal_state, :] = goal_reward

    for s in range(n_states):
        r, c = state_rc(s)

        if s == goal_state:
            # Goal is absorbing
            T[s, :, s] = 1.0
            continue

        for a in range(n_actions):
            # Intended move
            dr, dc = action_deltas[a]
            nr, nc = r + dr, c + dc

            # Clip to grid boundaries (wall = stay)
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                intended_next = state_idx(nr, nc)
            else:
                intended_next = s

            # With probability (1 - slip_prob), go to intended state
            T[s, a, intended_next] += (1 - slip_prob)

            # With probability slip_prob, go to a random adjacent cell
            # (uniformly over all 4 directions, clipped to boundaries)
            for a2 in range(n_actions):
                dr2, dc2 = action_deltas[a2]
                nr2, nc2 = r + dr2, c + dc2
                if 0 <= nr2 < grid_size and 0 <= nc2 < grid_size:
                    slip_next = state_idx(nr2, nc2)
                else:
                    slip_next = s
                T[s, a, slip_next] += slip_prob / n_actions

    # Verify row sums
    assert np.allclose(T.sum(axis=2), 1.0, atol=1e-10)

    return TabularMDP(n_states, n_actions, T, R, gamma)


# =============================================================================
# 3. RANDOMWORLD (15 states, 5 actions)
# =============================================================================

def make_randomworld(
    n_states: int = 15,
    n_actions: int = 5,
    gamma: float = 0.9,
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
) -> TabularMDP:
    """
    Randomly generated MDP matching the paper's RandomWorld benchmark.

    Transitions are sampled from Dirichlet(alpha, ..., alpha) for each (s, a).
    Lower alpha values produce sparser (more concentrated) transitions,
    which better represents real-world dynamics where states typically
    transition to a small subset of successors.

    Rewards are sampled from N(0, 1) for each (s, a).

    Args:
        n_states: number of states (paper uses 15)
        n_actions: number of actions (paper uses 5)
        gamma: discount factor
        dirichlet_alpha: concentration parameter (0.5 = sparse, 1.0 = uniform)
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Random transitions: lower alpha -> sparser transitions
    T = rng.dirichlet(np.full(n_states, dirichlet_alpha), size=(n_states, n_actions))

    # Random rewards
    R = rng.standard_normal((n_states, n_actions))

    return TabularMDP(n_states, n_actions, T, R, gamma)
