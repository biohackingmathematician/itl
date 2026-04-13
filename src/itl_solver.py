"""
Core ITL Solver — Inverse Transition Learning via Quadratic Programming.

Implements Algorithm 1 and Eq 10 from Benac et al. (2024), "Inverse Transition
Learning: Learning Dynamics from Demonstrations" (AAAI 2025).

Textbook cross-reference (Krause & Hubotter, "Probabilistic Artificial
Intelligence"):
  - Ch 10 (Tabular RL): policy evaluation v^pi = (I - gamma P^pi)^-1 r^pi
    (Eq 10.20) is the equation ITL linearizes and embeds as a constraint.
    Q*-function and Bellman optimality are Ch 10 Eq 10.9 / Def 10.7.
  - Ch 11 (Model-based Bayesian RL): motivates learning T* from finite counts
    and provides the Dirichlet-Categorical conjugate framework that MLE +
    Laplace smoothing (paper Eq 5) is the MAP case of. See docs/book_mapping.md.

Key fixes vs. the previous version (see results/MVR_findings.md):

  1. The epsilon-ball for constraints is now derived from OBSERVED expert
     actions at s in D (per the paper's definition), NOT from Q_hat under
     T_hat. The paper (page 4):
       "we construct an estimated policy pi_hat(.|s;T*), such that for s in
        D, pi_hat(a|s;T*) assigns a uniform probability to all actions a
        present in D"
     And Definition 1: E_eps(s;T*) is the set of actions the eps-optimal
     expert policy takes at s. We observe these directly from the data.

  2. Constraints are only added for (s, a) in D (visited state-action pairs),
     per the scope "for all (s, a) in D" in Eq 10.

  3. The linearization ALWAYS uses T_MLE (never T_hat): v_lin is computed
     with P^pi_hat = sum_a pi_hat[s,a] * T_MLE[s,a,s'], per the paper's
     linearization trick (page 4).

  4. Initial policy pi^(0)(.|s) for s NOT in D is UNIFORM (Algorithm 1 line 3),
     not greedy. After iteration 0, it switches to pi*(T^(i)) for s not in D.

  5. Loop termination: the paper stops when T satisfies the epsilon-ball
     property for each s in D. We check that explicitly, in addition to a
     safety max_iter and a tiny-change fallback.
"""

from __future__ import annotations

import warnings

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional, Dict

from .mdp import TabularMDP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_linearized_value(
    T_mle: np.ndarray,
    R: np.ndarray,
    pi_hat: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Paper's linearization trick (page 4): substitute T_MLE^{pi_hat} for T in
    (I - gamma * T^pi)^{-1}, making Eq 8/9 linear in the decision variable.

        v_lin = (I - gamma * T_MLE^{pi_hat})^{-1} R^{pi_hat}

    The bracketed operator is exactly the policy-evaluation linear system from
    Krause & Hubotter Ch 10 Eq 10.20 (v^pi = (I - gamma P^pi)^-1 r^pi), with
    T_MLE substituted for T so the solve is a fixed matrix at QP time.

    Returns:
        v_lin: (n_states,) linearized value vector
    """
    P_pi = np.einsum("sa,sab->sb", pi_hat, T_mle)
    r_pi = np.einsum("sa,sa->s", pi_hat, R)
    A = np.eye(T_mle.shape[0]) - gamma * P_pi
    return np.linalg.solve(A, r_pi)


def _initial_policy_from_data(
    visited_sa: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    """
    Algorithm 1, lines 2-3:
        pi^(0)(.|s) = uniform over observed actions  for s in D
        pi^(0)(.|s) = uniform over ALL actions       for s not in D
    """
    n_states = visited_sa.shape[0]
    pi = np.zeros((n_states, n_actions))
    for s in range(n_states):
        chosen = visited_sa[s]
        if chosen.any():
            pi[s, chosen] = 1.0 / chosen.sum()
        else:
            pi[s] = 1.0 / n_actions
    return pi


def _next_policy(
    visited_sa: np.ndarray,
    Q_hat: np.ndarray,
) -> np.ndarray:
    """
    Algorithm 1, lines 8-9:
        pi^(i)(.|s) = uniform over observed actions   for s in D
        pi^(i)(.|s) = one-hot on argmax_a Q(s,a;T^(i))  for s not in D
    """
    n_states, n_actions = visited_sa.shape
    pi = np.zeros((n_states, n_actions))
    for s in range(n_states):
        chosen = visited_sa[s]
        if chosen.any():
            pi[s, chosen] = 1.0 / chosen.sum()
        else:
            best_a = int(np.argmax(Q_hat[s]))
            pi[s, best_a] = 1.0
    return pi


def _epsilon_ball_matches_observed(
    T_hat: np.ndarray,
    R: np.ndarray,
    gamma: float,
    epsilon: float,
    visited_sa: np.ndarray,
) -> Tuple[bool, np.ndarray]:
    """
    Check whether T_hat's epsilon-ball at every observed state s matches
    the set of actions the expert was observed taking at s.

    Returns (satisfied, Q_hat).
    """
    mdp = TabularMDP(T_hat.shape[0], T_hat.shape[1], T_hat, R, gamma)
    _, Q_hat, _ = mdp.compute_optimal_policy()
    q_max = Q_hat.max(axis=1, keepdims=True)
    ball = (q_max - Q_hat) <= epsilon + 1e-9  # (n_states, n_actions)

    for s in range(visited_sa.shape[0]):
        if not visited_sa[s].any():
            continue
        # Paper requirement: eps-ball under T_hat at observed s == observed actions
        if not np.array_equal(ball[s], visited_sa[s]):
            return False, Q_hat
    return True, Q_hat


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_itl(
    N: np.ndarray,
    T_mle: np.ndarray,
    R: np.ndarray,
    gamma: float,
    epsilon: float,
    max_iter: int = 10,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Algorithm 1 from Benac et al. (2024).

    Args:
        N: (S, A, S) transition counts from the batch dataset
        T_mle: (S, A, S) Laplace-smoothed MLE transitions
        R: (S, A) rewards (assumed known)
        gamma: discount factor
        epsilon: near-optimality tolerance
        max_iter: safety cap on alternation iterations
        verbose: print per-iteration diagnostics

    Returns:
        T_hat: (S, A, S) estimated dynamics
        info: diagnostics dict
    """
    n_states, n_actions, _ = N.shape
    sa_counts = N.sum(axis=2)                 # (S, A) — #times expert picked a at s
    visited_sa = sa_counts > 0                # (S, A) — observed action mask
    visited_s = visited_sa.any(axis=1)        # (S,)   — was state ever visited

    info: Dict = {
        "iterations": [],
        "objective_values": [],
        "converged": False,
        "termination": "max_iter",
    }

    # --- Algorithm 1 lines 1-5: initialize and take first step ---------------
    pi_hat = _initial_policy_from_data(visited_sa, n_actions)
    T_hat = T_mle.copy()

    for iteration in range(max_iter):
        # Linearized value vector (paper Eq 10, using T_MLE, not T_hat)
        v_lin = compute_linearized_value(T_mle, R, pi_hat, gamma)

        # Solve inner QP
        T_new, obj_val, status = _solve_qp(
            N=N,
            T_mle=T_mle,
            R=R,
            gamma=gamma,
            epsilon=epsilon,
            v_lin=v_lin,
            visited_sa=visited_sa,
            visited_s=visited_s,
            n_states=n_states,
            n_actions=n_actions,
            verbose=verbose,
        )

        if T_new is None:
            info["iterations"].append({"status": "infeasible"})
            info["termination"] = "qp_failed"
            if verbose:
                print(f"  Iteration {iteration}: QP failed ({status})")
            break

        change = float(np.max(np.abs(T_new - T_hat)))
        info["iterations"].append(
            {"change": change, "objective": float(obj_val), "status": status}
        )
        info["objective_values"].append(float(obj_val))
        if verbose:
            print(
                f"  Iteration {iteration}: obj={obj_val:.6f}  "
                f"max_change={change:.6f}  status={status}"
            )

        T_hat = T_new

        # Paper termination: eps-ball property holds at every observed state
        satisfied, Q_hat = _epsilon_ball_matches_observed(
            T_hat, R, gamma, epsilon, visited_sa
        )
        if satisfied:
            info["converged"] = True
            info["termination"] = "eps_ball_property"
            if verbose:
                print(f"  eps-ball property satisfied; stopping.")
            break

        # Fallback: numerical fixed point
        if iteration > 0 and change < 1e-8:
            info["converged"] = True
            info["termination"] = "numerical_fixed_point"
            break

        # Update pi for next alternation (Algorithm 1 lines 8-9)
        pi_hat = _next_policy(visited_sa, Q_hat)

    return T_hat, info


# ---------------------------------------------------------------------------
# Inner QP
# ---------------------------------------------------------------------------


def _solve_qp(
    N: np.ndarray,
    T_mle: np.ndarray,
    R: np.ndarray,
    gamma: float,
    epsilon: float,
    v_lin: np.ndarray,
    visited_sa: np.ndarray,
    visited_s: np.ndarray,
    n_states: int,
    n_actions: int,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    """
    Solve the convex QP in Eq 10 of Benac et al. (2024):

        min_T  sum_{s,a,s'} N[s,a,s'] * (T[s,a,s'] - T_MLE[s,a,s'])^2

        s.t. for all s in D, all a in observed(s), all a' NOT in observed(s):
                R(s,a) - R(s,a') + gamma (T_sa - T_sa')^T v_lin >= epsilon   (Eq 8)
             for all s in D, all a, a' in observed(s), a != a':
                |R(s,a) - R(s,a') + gamma (T_sa - T_sa')^T v_lin| <= epsilon  (Eq 9)
             T[s,a,s'] >= 0,  sum_{s'} T[s,a,s'] = 1 for all (s, a).

    The expression R(s,a) + gamma * T_sa^T v_lin is the Q-function
    (Krause & Hubotter Ch 10 Eq 10.9) evaluated under the linearized value
    v_lin; the constraints encode the eps-optimality condition Q*(s,a) is
    within eps of max_a' Q*(s,a').
    """
    # Decision variables (one simplex per (s, a))
    T_var = np.empty((n_states, n_actions), dtype=object)
    for s in range(n_states):
        for a in range(n_actions):
            T_var[s, a] = cp.Variable(n_states, nonneg=True)

    # Weighted L2 objective
    obj_terms = []
    for s in range(n_states):
        for a in range(n_actions):
            w = N[s, a]  # (S,)
            if w.sum() == 0:
                # Unobserved (s, a): pull toward T_MLE very weakly so variables
                # don't float off into solver noise. Tiny weight, does not
                # affect constraints.
                obj_terms.append(
                    1e-6 * cp.sum_squares(T_var[s, a] - T_mle[s, a])
                )
            else:
                diff = T_var[s, a] - T_mle[s, a]
                obj_terms.append(cp.sum(cp.multiply(w, cp.square(diff))))

    objective = cp.Minimize(cp.sum(obj_terms))

    # Constraints
    constraints = []

    # Simplex constraints for every (s, a)
    for s in range(n_states):
        for a in range(n_actions):
            constraints.append(cp.sum(T_var[s, a]) == 1)

    # ITL constraints ONLY at observed states, using observed actions as valid
    for s in range(n_states):
        if not visited_s[s]:
            continue
        valid = np.where(visited_sa[s])[0]
        invalid = np.where(~visited_sa[s])[0]

        # Eq 8: each observed action >= epsilon-better than each unobserved action
        for a in valid:
            for a_bad in invalid:
                dR = R[s, a] - R[s, a_bad]
                dT = T_var[s, a] - T_var[s, a_bad]
                constraints.append(dR + gamma * dT @ v_lin >= epsilon)

        # Eq 9: observed actions are within epsilon of each other
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, a2 = int(valid[i]), int(valid[j])
                dR = R[s, a] - R[s, a2]
                dT = T_var[s, a] - T_var[s, a2]
                expr = dR + gamma * dT @ v_lin
                constraints.append(expr <= epsilon)
                constraints.append(expr >= -epsilon)

    # Solve, try several solvers before giving up
    problem = cp.Problem(objective, constraints)
    status = ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for solver_kwargs in (
            dict(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8,
                 max_iter=100000, polish=True),
            dict(solver=cp.SCS, eps=1e-8, max_iters=100000),
            dict(solver=cp.CLARABEL) if hasattr(cp, "CLARABEL") else None,
        ):
            if solver_kwargs is None:
                continue
            try:
                problem.solve(verbose=False, **solver_kwargs)
                status = str(problem.status)
                if status in ("optimal", "optimal_inaccurate"):
                    break
            except (cp.SolverError, Exception):
                status = "solver_error"
                continue

    if problem.status not in ("optimal", "optimal_inaccurate"):
        if verbose:
            print(f"    QP status: {problem.status}")
        return None, None, str(problem.status)

    # Extract solution; project onto simplex as a safety net
    T_hat = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            val = T_var[s, a].value
            if val is None:
                T_hat[s, a] = T_mle[s, a]
            else:
                val = np.maximum(val, 0.0)
                total = val.sum()
                if total > 1e-12:
                    val = val / total
                else:
                    val = T_mle[s, a].copy()
                T_hat[s, a] = val

    return T_hat, float(problem.value), str(problem.status)
