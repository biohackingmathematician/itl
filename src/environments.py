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
# 2b. TWO-GOAL GRIDWORLD (non-goal-dominated benchmark for ITL+IRL)
# =============================================================================
#
# Motivation: the standard Gridworld is "goal-dominated" — pi*(T_MLE,
# R_trivial = 10 * 1[s == goal]) matches pi*(T_MLE, R_TRUE) almost
# perfectly because all the relevant policy information is "find the
# unique goal". A reward learner that only locates the goal looks
# indistinguishable from one that recovers the full reward structure.
#
# The two-goal benchmark fixes this: with two competing absorbing goals
# of *different* reward magnitudes (R_A < R_B), the optimal policy
# depends on the magnitude difference, not just the goal locations. A
# trivial reward "+10 at both goals" cannot reproduce pi* because it
# can't tell A and B apart, so the agent picks whichever is closer.
#
# See `docs/c_itl_options.md` "Methodology gap discovered 2026-05-01"
# for the full motivation and the falsifier this benchmark certifies.


def make_two_goal_gridworld(
    grid_size: int = 5,
    gamma: float = 0.95,
    slip_prob: float = 0.2,
    R_A: float = 5.0,
    R_B: float = 10.0,
    step_cost: float = -0.1,
    soft_wall_penalty: float = -5.0,
    soft_walls: Optional[List[Tuple[int, int]]] = None,
) -> TabularMDP:
    """
    Gridworld with two competing absorbing goals of different reward
    magnitudes, separated by a soft-wall barrier. Designed to be NOT
    goal-dominated, so that recovering the reward STRUCTURE (not just
    the goal locations) is required to match pi*.

    States: ``grid_size * grid_size`` cells, indexed ``row * grid_size + col``.
    Actions: 0=Up, 1=Down, 2=Left, 3=Right (matches `make_gridworld`).
    Goal A: top-right ``(0, grid_size - 1)``, absorbing, reward ``R_A``.
    Goal B: bottom-left ``(grid_size - 1, 0)``, absorbing, reward ``R_B``.
    Step cost on every non-terminal, non-barrier tile.
    Soft-wall penalty on barrier tiles (the agent still transitions; the
    penalty is the reward for *entering* the tile).
    Slip mechanics identical to `make_gridworld`.

    Args:
        grid_size: side length of the square grid (default 5).
        gamma: discount factor (default 0.95, paper Gridworld value).
        slip_prob: probability of slipping to a neighbor of the intended
            destination (default 0.2, paper value).
        R_A: reward at the smaller goal A (default 5.0).
        R_B: reward at the larger goal B (default 10.0); B is the policy-
            relevant goal in expectation when the barrier doesn't make
            it too expensive.
        step_cost: per-step reward on non-terminal, non-barrier tiles.
        soft_wall_penalty: per-step reward on barrier tiles.
        soft_walls: optional list of (row, col) for the barrier. If
            None, defaults to the main-diagonal cells excluding the
            corners — `[(i, i) for i in range(1, grid_size - 1)]` —
            which forces direct anti-diagonal trajectories between A
            and B to either eat the penalty or detour around the
            corners. The default is calibrated to make the env
            non-goal-dominated for `R_A=5`, `R_B=10`, slip 0.2; if you
            change those, re-verify with
            `tests.test_smoke.test_two_goal_is_NOT_goal_dominated`.

    Returns:
        TabularMDP with shape (S=grid_size**2, A=4, S=grid_size**2).
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

    goal_A = state_idx(0, grid_size - 1)            # top-right
    goal_B = state_idx(grid_size - 1, 0)            # bottom-left

    if soft_walls is None:
        # Default: a "thick diagonal" barrier — main-diagonal cells plus the
        # cells on either side of the centre cell — which blocks both the
        # exact anti-diagonal and the immediately adjacent shortcuts. The
        # thinner barrier (just the main diagonal minus corners) gives a
        # mean truth-vs-trivial gap right at the 0.10 threshold and
        # occasionally falls below it; this thicker layout pushes the gap
        # consistently into the 0.15–0.20 range across seeds 0..4 with
        # `R_A=5`, `R_B=10`, slip 0.2, step cost -0.1, penalty -5.
        if grid_size == 5:
            soft_walls = [
                (1, 1), (2, 2), (3, 3),     # main diagonal (no corners)
                (1, 2), (2, 1),              # cells flanking (2, 2) on the
                (2, 3), (3, 2),              # near-anti-diagonal directions
            ]
        else:
            # General-grid fallback: just the main-diagonal cells minus
            # corners. Re-tune for a different grid size if needed.
            soft_walls = [(i, i) for i in range(1, grid_size - 1)]
    soft_wall_set = {state_idx(r, c) for (r, c) in soft_walls}

    if goal_A in soft_wall_set or goal_B in soft_wall_set:
        raise ValueError(
            "soft_walls must not include either goal cell; got "
            f"goal_A={goal_A}, goal_B={goal_B}, walls={sorted(soft_wall_set)}"
        )

    T = np.zeros((n_states, n_actions, n_states))
    R = np.full((n_states, n_actions), step_cost)

    R[goal_A, :] = R_A
    R[goal_B, :] = R_B
    for sw in soft_wall_set:
        R[sw, :] = soft_wall_penalty

    def clip(r, c):
        if 0 <= r < grid_size and 0 <= c < grid_size:
            return r, c
        return None

    for s in range(n_states):
        r, c = state_rc(s)

        if s == goal_A or s == goal_B:
            T[s, :, s] = 1.0
            continue

        for a in range(n_actions):
            dr, dc = action_deltas[a]
            intended_rc = clip(r + dr, c + dc)
            if intended_rc is None:
                intended_next = s
                ir, ic = r, c
            else:
                ir, ic = intended_rc
                intended_next = state_idx(ir, ic)

            T[s, a, intended_next] += (1 - slip_prob)

            for (ddr, ddc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                slip_rc = clip(ir + ddr, ic + ddc)
                if slip_rc is None:
                    slip_next = intended_next
                else:
                    slip_next = state_idx(*slip_rc)
                T[s, a, slip_next] += slip_prob / 4.0

    T /= T.sum(axis=2, keepdims=True)
    return TabularMDP(n_states, n_actions, T, R, gamma)


def two_goal_states(grid_size: int = 5) -> Tuple[int, int]:
    """Return (goal_A_state, goal_B_state) for the two-goal env."""
    goal_A = 0 * grid_size + (grid_size - 1)
    goal_B = (grid_size - 1) * grid_size + 0
    return goal_A, goal_B


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
