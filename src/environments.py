"""
Environment definitions for ITL reproduction.

Matches the specification in Benac et al. (2024), Appendix "Synthetic Environments":

  1. Corridor (3-state) — toy MDP from Agna's notes, for verification
  2. Gridworld (5x5, 25 states, 4 actions) — paper's flagship synthetic benchmark,
     with soft-wall tiles and start/goal matching the paper figure
  3. RandomWorld (15 states, 5 actions) — 5 successors per (s,a) drawn from
     Uniform[0,1] and normalized; state-only rewards R(s) ~ Uniform[16-s-1, 16-s]
"""

import numpy as np
from typing import Optional, Tuple, List
from .mdp import TabularMDP


# =============================================================================
# 1. CORRIDOR (3-state toy MDP, unchanged — used for hand-calc verification)
# =============================================================================

def make_corridor(gamma: float = 0.9) -> TabularMDP:
    """
    3-state corridor from the March 2026 worked example.

    States: s1, s2, s_goal (indices 0, 1, 2)
    Actions: Left=0, Right=1
    Hand-computed (gamma=0.9): v* = [76.87, 87.68, 100.00]

    Note: uses gamma=0.9 by default to match hand calcs, not the paper's 0.95.
    """
    n_states, n_actions = 3, 2

    T = np.zeros((n_states, n_actions, n_states))
    T[0, 1] = [0.2, 0.8, 0.0]
    T[0, 0] = [1.0, 0.0, 0.0]
    T[1, 1] = [0.0, 0.2, 0.8]
    T[1, 0] = [0.8, 0.2, 0.0]
    T[2, 0] = [0.0, 0.0, 1.0]
    T[2, 1] = [0.0, 0.0, 1.0]

    R = np.array([
        [-0.1, -0.1],
        [-0.1, -0.1],
        [10.0, 10.0],
    ])

    return TabularMDP(n_states, n_actions, T, R, gamma)


# =============================================================================
# 2. GRIDWORLD (5x5, paper spec)
# =============================================================================

# Soft-wall positions for the standard task, roughly matching the paper's Figure 5.
# Paper shows a "diagonal barrier" of soft-wall tiles forcing the optimal path
# to curve around them. Exact positions aren't published, so we use a diagonal
# barrier that forces a non-trivial optimal policy.
DEFAULT_SOFT_WALLS_STANDARD = [(1, 1), (1, 2), (2, 1), (3, 3), (3, 2)]

# Transfer task relocates the barrier to force a different optimal path.
DEFAULT_SOFT_WALLS_TRANSFER = [(1, 3), (2, 3), (2, 2), (3, 1), (3, 2)]


def make_gridworld(
    grid_size: int = 5,
    gamma: float = 0.95,
    goal_reward: float = 10.0,
    step_cost: float = -0.1,
    soft_wall_penalty: float = -5.0,
    slip_prob: float = 0.2,
    soft_walls: Optional[List[Tuple[int, int]]] = None,
    transfer: bool = False,
) -> TabularMDP:
    """
    5x5 Gridworld matching Benac et al. (2024), Appendix "Gridworld Environment".

    States: 25 cells, indexed row * grid_size + col.
    Actions: 0=Up, 1=Down, 2=Left, 3=Right.
    Start: bottom-left (row=grid_size-1, col=0), not encoded in MDP itself but
           used by the experiment driver for episode initialization.
    Goal: top-right (row=0, col=grid_size-1), absorbing with reward +10.
    Step cost: -0.1 on normal tiles.
    Soft walls: -5 on designated tiles (the agent still transitions; the -5 is
                the reward for *entering* the tile).
    Slip: intended move with prob (1 - slip_prob) = 0.8. With prob slip_prob,
          land in one of the 4 cells adjacent to the *intended* destination,
          uniformly at random, clipped to grid boundaries.

    The transfer task uses a different soft-wall layout (same grid, different R).
    """
    n_states = grid_size * grid_size
    n_actions = 4

    def state_idx(r, c):
        return r * grid_size + c

    def state_rc(s):
        return s // grid_size, s % grid_size

    action_deltas = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }

    goal_state = state_idx(0, grid_size - 1)

    if soft_walls is None:
        soft_walls = DEFAULT_SOFT_WALLS_TRANSFER if transfer else DEFAULT_SOFT_WALLS_STANDARD
    soft_wall_set = {state_idx(r, c) for (r, c) in soft_walls}

    T = np.zeros((n_states, n_actions, n_states))
    R = np.full((n_states, n_actions), step_cost)

    def clip(r, c):
        """Return (r', c') after clipping to grid; if off-grid, stay put."""
        if 0 <= r < grid_size and 0 <= c < grid_size:
            return r, c
        return None

    for s in range(n_states):
        r, c = state_rc(s)

        # Reward on entering state s: goal / soft-wall / step
        if s == goal_state:
            R[s, :] = goal_reward
        elif s in soft_wall_set:
            R[s, :] = soft_wall_penalty

        if s == goal_state:
            # Absorbing
            T[s, :, s] = 1.0
            continue

        for a in range(n_actions):
            dr, dc = action_deltas[a]
            intended_rc = clip(r + dr, c + dc)
            if intended_rc is None:
                intended_next = s  # bump into wall -> stay
                ir, ic = r, c
            else:
                ir, ic = intended_rc
                intended_next = state_idx(ir, ic)

            # With prob 1 - slip_prob, arrive at intended
            T[s, a, intended_next] += (1 - slip_prob)

            # With prob slip_prob, land on one of the 4 neighbors of the
            # INTENDED destination (paper: "slip to any of the four neighboring
            # tiles of the intended state"), chosen uniformly.
            for (ddr, ddc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                slip_rc = clip(ir + ddr, ic + ddc)
                if slip_rc is None:
                    slip_next = intended_next  # off-grid -> stay at intended
                else:
                    slip_next = state_idx(*slip_rc)
                T[s, a, slip_next] += slip_prob / 4.0

    # Normalize row sums to exactly 1 (floating-point safety).
    T /= T.sum(axis=2, keepdims=True)

    return TabularMDP(n_states, n_actions, T, R, gamma)


def gridworld_start_state(grid_size: int = 5) -> int:
    """Bottom-left corner, per paper."""
    return (grid_size - 1) * grid_size + 0


# =============================================================================
# 3. RANDOMWORLD (paper spec)
# =============================================================================

def make_randomworld(
    n_states: int = 15,
    n_actions: int = 5,
    gamma: float = 0.95,
    n_successors: int = 5,
    seed: int = 42,
) -> TabularMDP:
    """
    Random MDP following Benac et al. (2024), Appendix "Randomworld Environment":

      - 15 states, 5 actions
      - For each (s, a): pick `n_successors` = 5 successor states uniformly at
        random (without replacement), draw their probabilities independently
        from Uniform[0, 1], and normalize to sum to 1.
      - Rewards depend on STATE only (not action): R(s) ~ Uniform[16-s-1, 16-s],
        so state 0 (labeled 1 in paper) is best (Unif[15,16]) and state 14
        (labeled 15 in paper) is worst (Unif[0,1]).
    """
    rng = np.random.default_rng(seed)

    T = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            successors = rng.choice(n_states, size=n_successors, replace=False)
            probs = rng.uniform(0.0, 1.0, size=n_successors)
            probs /= probs.sum()
            T[s, a, successors] = probs

    # State-only rewards: state s (0-indexed) corresponds to paper's state s+1.
    # Paper: R(s_paper) ~ Uniform[16 - s_paper - 1, 16 - s_paper]
    # With 0-indexed s = s_paper - 1: R(s) ~ Uniform[15 - s - 1, 15 - s] = Uniform[14-s, 15-s]
    R_state = np.array([rng.uniform(15 - s - 1, 15 - s) for s in range(n_states)])
    R = np.tile(R_state[:, None], (1, n_actions))

    return TabularMDP(n_states, n_actions, T, R, gamma)


def make_randomworld_transfer(mdp: TabularMDP, seed: int = 0) -> TabularMDP:
    """
    Transfer task for RandomWorld: same T, reversed reward structure.
    Paper: R_transfer(s_paper) ~ Uniform[s_paper - 1, s_paper].
    With 0-indexed s: R(s) ~ Uniform[s, s+1].
    """
    rng = np.random.default_rng(seed)
    n_states = mdp.n_states
    n_actions = mdp.n_actions
    R_state = np.array([rng.uniform(s, s + 1) for s in range(n_states)])
    R = np.tile(R_state[:, None], (1, n_actions))
    return TabularMDP(n_states, n_actions, mdp.T, R, mdp.gamma)
