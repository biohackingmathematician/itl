"""
Core ITL Solver — Inverse Transition Learning via Quadratic Programming.

Implements Algorithm 1 from Benac et al. (2024).

The optimization problem (Eq 10 in the paper):

    min_T  sum_{s,a,s'} N_{s,a,s'} * [T(s'|s,a) - T_mle(s'|s,a)]^2

    subject to:
        Constraint 1 (Eq 8): For valid a, invalid a':
            R(s,a) - R(s,a') + gamma * [T(.|s,a) - T(.|s,a')]^T * v_lin >= epsilon

        Constraint 2 (Eq 9): For valid a, valid a':
            |R(s,a) - R(s,a') + gamma * [T(.|s,a) - T(.|s,a')]^T * v_lin| <= epsilon

        Plus: T(s'|s,a) >= 0, sum_{s'} T(s'|s,a) = 1 for all (s,a)

    where v_lin = (I - gamma * T^pi_mle)^{-1} R^pi
    is the linearized value vector (the "linearization trick").
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional
from .mdp import TabularMDP


def compute_linearized_value(
    T_mle: np.ndarray,
    R: np.ndarray,
    pi_hat: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute the linearized value vector used in ITL's constraints.

    v_lin = (I - gamma * T^{pi_hat}_mle)^{-1} * R^{pi_hat}

    This is the "linearization trick" (Section 3.2 of the paper):
    we substitute T_mle under pi_hat for the unknown T^pi inside the
    matrix inverse, making constraints linear in the decision variable T.

    Args:
        T_mle: MLE transitions, shape (n_states, n_actions, n_states)
        R: rewards, shape (n_states, n_actions)
        pi_hat: estimated expert policy, shape (n_states, n_actions)
        gamma: discount factor

    Returns:
        v_lin: linearized value vector, shape (n_states,)
    """
    n_states = T_mle.shape[0]

    # P^pi_mle[s, s'] = sum_a pi_hat[s,a] * T_mle[s,a,s']
    P_pi = np.einsum("sa,sab->sb", pi_hat, T_mle)

    # r^pi[s] = sum_a pi_hat[s,a] * R[s,a]
    r_pi = np.einsum("sa,sa->s", pi_hat, R)

    # v = (I - gamma * P^pi)^{-1} r^pi
    A = np.eye(n_states) - gamma * P_pi
    v_lin = np.linalg.solve(A, r_pi)

    return v_lin


def estimate_expert_policy(
    N: np.ndarray,
    T_current: np.ndarray,
    R: np.ndarray,
    gamma: float,
    epsilon: float,
) -> np.ndarray:
    """
    Estimate the expert policy pi_hat from data (Algorithm 1, line 6-9).

    For visited (s, a) pairs: uniform over actions the expert actually chose.
    For unvisited states: use optimal action under current T estimate.

    Args:
        N: transition counts, shape (n_states, n_actions, n_states)
        T_current: current transition estimate, shape (n_states, n_actions, n_states)
        R: rewards, shape (n_states, n_actions)
        gamma: discount factor
        epsilon: near-optimality tolerance

    Returns:
        pi_hat: estimated expert policy, shape (n_states, n_actions)
    """
    n_states, n_actions, _ = N.shape
    pi_hat = np.zeros((n_states, n_actions))

    # Which (s, a) pairs were visited?
    sa_counts = N.sum(axis=2)  # (n_states, n_actions)
    visited_states = sa_counts.sum(axis=1) > 0  # (n_states,)

    for s in range(n_states):
        if visited_states[s]:
            # Uniform over actions the expert chose at this state
            actions_chosen = sa_counts[s] > 0
            n_chosen = actions_chosen.sum()
            if n_chosen > 0:
                pi_hat[s, actions_chosen] = 1.0 / n_chosen
            else:
                pi_hat[s] = 1.0 / n_actions
        else:
            # Unvisited state: use greedy action under current T
            # Build a temporary MDP with T_current to find optimal action
            temp_mdp = TabularMDP(n_states, n_actions, T_current, R, gamma)
            _, Q, _ = temp_mdp.compute_optimal_policy()
            best_a = Q[s].argmax()
            pi_hat[s, best_a] = 1.0

    return pi_hat


def solve_itl(
    N: np.ndarray,
    T_mle: np.ndarray,
    R: np.ndarray,
    gamma: float,
    epsilon: float,
    max_iter: int = 10,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Solve the ITL optimization problem (Algorithm 1).

    Alternates between:
      1. Estimating expert policy pi_hat given current T
      2. Solving the QP for T given pi_hat and linearized value

    Args:
        N: transition counts from data, shape (n_states, n_actions, n_states)
        T_mle: MLE transition estimate, shape (n_states, n_actions, n_states)
        R: known reward function, shape (n_states, n_actions)
        gamma: discount factor
        epsilon: near-optimality tolerance
        max_iter: maximum alternation iterations
        verbose: print progress

    Returns:
        T_hat: estimated transition matrix, shape (n_states, n_actions, n_states)
        info: dict with convergence info, diagnostics
    """
    n_states, n_actions, _ = N.shape

    # Initialize T_hat with MLE (or uniform for unvisited)
    T_hat = T_mle.copy()

    info = {"iterations": [], "objective_values": [], "converged": False}

    for iteration in range(max_iter):
        # Step 1: Estimate expert policy
        pi_hat = estimate_expert_policy(N, T_hat, R, gamma, epsilon)

        # Step 2: Compute linearized value vector
        v_lin = compute_linearized_value(T_mle, R, pi_hat, gamma)

        # Step 3: Determine epsilon-ball (which actions are valid/invalid)
        # Use current T_hat to compute Q-values and classify actions
        temp_mdp = TabularMDP(n_states, n_actions, T_hat, R, gamma)
        _, Q_hat, _ = temp_mdp.compute_optimal_policy()
        valid = temp_mdp.compute_epsilon_ball(Q_hat, epsilon)

        # Step 4: Solve the QP
        T_new, obj_val = _solve_qp(
            N, T_mle, R, gamma, epsilon, v_lin, valid, n_states, n_actions, verbose
        )

        if T_new is None:
            if verbose:
                print(f"  Iteration {iteration}: QP infeasible")
            info["iterations"].append({"status": "infeasible"})
            break

        # Check convergence
        change = np.max(np.abs(T_new - T_hat))
        info["iterations"].append({
            "change": change,
            "objective": obj_val,
            "status": "solved",
        })
        info["objective_values"].append(obj_val)

        if verbose:
            print(f"  Iteration {iteration}: obj={obj_val:.6f}, max_change={change:.6f}")

        if change < 1e-6:
            info["converged"] = True
            T_hat = T_new
            break

        T_hat = T_new

    return T_hat, info


def _solve_qp(
    N: np.ndarray,
    T_mle: np.ndarray,
    R: np.ndarray,
    gamma: float,
    epsilon: float,
    v_lin: np.ndarray,
    valid: np.ndarray,
    n_states: int,
    n_actions: int,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Solve the inner QP (Eq 10) for a fixed linearization point.

    Decision variable: T[s, a, s'] for all (s, a, s')

    Objective:
        min sum_{s,a,s'} N_{s,a,s'} * (T[s,a,s'] - T_mle[s,a,s'])^2

    Constraints:
        1. For valid a, invalid a' at state s:
           R(s,a) - R(s,a') + gamma * (T(.|s,a) - T(.|s,a'))^T v_lin >= epsilon
        2. For valid a, a' at state s:
           |R(s,a) - R(s,a') + gamma * (T(.|s,a) - T(.|s,a'))^T v_lin| <= epsilon
        3. T(s'|s,a) >= 0 for all s, a, s'
        4. sum_{s'} T(s'|s,a) = 1 for all s, a
    """
    # Decision variable: one variable per (s, a, s') entry
    T_var = {}
    for s in range(n_states):
        for a in range(n_actions):
            T_var[s, a] = cp.Variable(n_states, nonneg=True)

    # Objective: weighted squared distance to MLE
    objective_terms = []
    for s in range(n_states):
        for a in range(n_actions):
            weights = N[s, a]  # (n_states,) — count for each s'
            diff = T_var[s, a] - T_mle[s, a]
            # Weighted squared L2: sum_s' N_{s,a,s'} * (T - T_mle)^2
            objective_terms.append(cp.sum(cp.multiply(weights, cp.square(diff))))

    objective = cp.Minimize(cp.sum(objective_terms))

    # Constraints
    constraints = []

    # Simplex constraints: sum to 1 for each (s, a)
    for s in range(n_states):
        for a in range(n_actions):
            constraints.append(cp.sum(T_var[s, a]) == 1)

    # ITL constraints (Eq 8 and 9)
    for s in range(n_states):
        valid_actions = np.where(valid[s])[0]
        invalid_actions = np.where(~valid[s])[0]

        for a_valid in valid_actions:
            # Constraint 1: valid vs invalid (Eq 8)
            for a_invalid in invalid_actions:
                reward_diff = R[s, a_valid] - R[s, a_invalid]
                transition_diff = T_var[s, a_valid] - T_var[s, a_invalid]
                lhs = reward_diff + gamma * transition_diff @ v_lin
                constraints.append(lhs >= epsilon)

            # Constraint 2: valid vs valid (Eq 9)
            for a_valid2 in valid_actions:
                if a_valid < a_valid2:  # avoid duplicate pairs
                    reward_diff = R[s, a_valid] - R[s, a_valid2]
                    transition_diff = T_var[s, a_valid] - T_var[s, a_valid2]
                    expr = reward_diff + gamma * transition_diff @ v_lin
                    constraints.append(expr <= epsilon)
                    constraints.append(expr >= -epsilon)

    # Solve
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except cp.SolverError:
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            return None, None

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        if verbose:
            print(f"    QP status: {problem.status}")
        return None, None

    # Extract solution
    T_hat = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            val = T_var[s, a].value
            if val is not None:
                # Project to simplex (clip negatives, renormalize)
                val = np.maximum(val, 0)
                val /= val.sum() + 1e-12
                T_hat[s, a] = val
            else:
                T_hat[s, a] = T_mle[s, a]

    return T_hat, problem.value
