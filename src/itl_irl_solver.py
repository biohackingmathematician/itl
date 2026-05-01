"""
Combined ITL + IRL solver — joint inference of (T*, R*) from eps-optimal
expert demonstrations.

This is the Direction 3 prototype from `docs/c_itl_options.md`. It extends
the paper's Algorithm 1 by lifting R from a known input to a decision
variable parameterized by user-provided reward features:

    R(s, a) = Phi(s, a) @ w           with w in R^d

The eps-ball constraints (Eq 8 / Eq 9 of Benac et al. 2024) become

    Eq 8: (Phi[s,a] - Phi[s,a'])^T w + gamma (T_sa - T_sa')^T v_lin >= eps
    Eq 9: |(Phi[s,a] - Phi[s,a'])^T w + gamma (T_sa - T_sa')^T v_lin| <= eps

These are LINEAR in (w, T) with v_lin treated as a fixed parameter (computed
at the previous outer iteration under T_MLE, the current pi_hat, and the
current w_hat). So the inner problem is a standard convex QP and we
alternate the same way Algorithm 1 alternates over policy.

Identifiability note:
- Without an anchor, w is identifiable only up to additive constant in
  any direction parallel to ker(Phi[s,a] - Phi[s,a']) for all (s, a, a').
- For state-only one-hot features (Phi[s,a,k] = 1[s == k]), all
  pairwise differences within a state are zero, so the eps-ball
  constraints don't pin down w. We rely on the L1 penalty + the
  inter-state Q comparisons indirectly via T to stabilize w.
- A practical anchor: fix w[0] = R_known or add a known reward at one
  reference (s, a). The corridor smoke test below uses the goal reward
  (+10) at state 2 as the anchor.

Status: PROTOTYPE. Smoke-tested on corridor with state-only features.
Not yet stress-tested on gridworld or randomworld. See task #19 for
follow-ups.
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


def _v_lin_under(T_mle, R_now, pi_hat, gamma):
    """v_lin under T_MLE^{pi_hat} with current reward estimate R_now."""
    P_pi = np.einsum("sa,sab->sb", pi_hat, T_mle)
    r_pi = np.einsum("sa,sa->s", pi_hat, R_now)
    A = np.eye(T_mle.shape[0]) - gamma * P_pi
    return np.linalg.solve(A, r_pi)


def _initial_policy_from_data(visited_sa, n_actions):
    n_states = visited_sa.shape[0]
    pi = np.zeros((n_states, n_actions))
    for s in range(n_states):
        chosen = visited_sa[s]
        if chosen.any():
            pi[s, chosen] = 1.0 / chosen.sum()
        else:
            pi[s] = 1.0 / n_actions
    return pi


def _next_policy(visited_sa, Q_hat):
    n_states, n_actions = visited_sa.shape
    pi = np.zeros((n_states, n_actions))
    for s in range(n_states):
        chosen = visited_sa[s]
        if chosen.any():
            pi[s, chosen] = 1.0 / chosen.sum()
        else:
            pi[s, int(np.argmax(Q_hat[s]))] = 1.0
    return pi


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_itl_irl(
    N: np.ndarray,
    T_mle: np.ndarray,
    Phi: np.ndarray,
    gamma: float,
    epsilon: float,
    w_init: Optional[np.ndarray] = None,
    anchor: Optional[Tuple[int, int, float]] = None,
    lambda_l1: float = 0.0,
    max_iter: int = 10,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Joint inference of T and R via alternating QP.

    Args:
        N: (S, A, S) transition counts from batch dataset
        T_mle: (S, A, S) Laplace-smoothed MLE transitions
        Phi: (S, A, d) reward feature tensor; R(s, a) = Phi(s, a) @ w
        gamma: discount factor
        epsilon: near-optimality tolerance
        w_init: initial reward weights, shape (d,). If None, zeros.
        anchor: optional triple (s, a, value) fixing R(s, a) = value. This
            breaks the additive-constant ambiguity in w.
        lambda_l1: L1 sparsity penalty on w (for identifiability).
        max_iter: outer alternation cap.
        tol: convergence tolerance on |w_new - w_prev|_inf.
        verbose: print per-iteration diagnostics.

    Returns:
        T_hat: (S, A, S) estimated dynamics
        w_hat: (d,) estimated reward weights; R_hat = Phi @ w_hat
        info: dict with iteration history
    """
    n_states, n_actions, _ = N.shape
    d = Phi.shape[2]
    assert Phi.shape == (n_states, n_actions, d), \
        f"Phi shape {Phi.shape} != expected ({n_states}, {n_actions}, {d})"

    sa_counts = N.sum(axis=2)
    visited_sa = sa_counts > 0
    visited_s = visited_sa.any(axis=1)

    info: Dict = {"iterations": [], "converged": False}

    pi_hat = _initial_policy_from_data(visited_sa, n_actions)
    T_hat = T_mle.copy()

    # If no w_init given, derive a sensible non-zero start. Without this, the
    # initial v_lin = 0 makes the eps-ball constraints infeasible at iter 0
    # for any feature scheme where Phi[s,a] - Phi[s,a'] = 0 within a state
    # (e.g., state-only features). Two strategies:
    #   1. If anchor is provided, satisfy it with a min-norm w (least-squares
    #      against the single anchor equation Phi[s_a, a_a] @ w = val).
    #   2. Otherwise, set every component of w to small random values; the
    #      L1 penalty + alternation will reshape it from there.
    if w_init is not None:
        w_hat = w_init.copy()
    elif anchor is not None:
        s_a, a_a, val = anchor
        phi_anchor = Phi[s_a, a_a]
        nrm_sq = float(phi_anchor @ phi_anchor)
        if nrm_sq > 1e-12:
            w_hat = (val / nrm_sq) * phi_anchor
        else:
            w_hat = np.zeros(d)
    else:
        rng_init = np.random.default_rng(0)
        w_hat = rng_init.normal(scale=0.1, size=d)

    for it in range(max_iter):
        # Linearized value vector (paper Eq 10 trick) using current (R, pi)
        R_now = Phi @ w_hat                                  # (S, A)
        v_lin = _v_lin_under(T_mle, R_now, pi_hat, gamma)    # (S,)

        # Solve joint QP for (T_new, w_new)
        T_new, w_new, obj_val, status = _solve_qp(
            N=N, T_mle=T_mle, Phi=Phi, gamma=gamma, epsilon=epsilon,
            v_lin=v_lin, visited_sa=visited_sa, visited_s=visited_s,
            n_states=n_states, n_actions=n_actions, d=d,
            anchor=anchor, lambda_l1=lambda_l1, verbose=verbose,
        )

        if T_new is None:
            info["iterations"].append({"status": "infeasible"})
            info["termination"] = "qp_failed"
            break

        w_change = float(np.max(np.abs(w_new - w_hat)))
        T_change = float(np.max(np.abs(T_new - T_hat)))
        info["iterations"].append({
            "obj": obj_val, "w_change": w_change, "T_change": T_change,
            "status": status, "w": w_new.tolist(),
        })
        if verbose:
            print(f"  iter {it}: obj={obj_val:.4f}  "
                  f"|dw|_inf={w_change:.4f}  |dT|_inf={T_change:.4f}")

        T_hat = T_new
        w_hat = w_new

        # Update pi_hat for next iteration
        R_hat = Phi @ w_hat
        mdp_now = TabularMDP(n_states, n_actions, T_hat, R_hat, gamma)
        _, Q_hat, _ = mdp_now.compute_optimal_policy()
        pi_hat = _next_policy(visited_sa, Q_hat)

        if it > 0 and w_change < tol and T_change < tol:
            info["converged"] = True
            info["termination"] = "fixed_point"
            break

    if not info.get("termination"):
        info["termination"] = "max_iter"

    return T_hat, w_hat, info


# ---------------------------------------------------------------------------
# Inner QP
# ---------------------------------------------------------------------------


def _solve_qp(
    N, T_mle, Phi, gamma, epsilon, v_lin, visited_sa, visited_s,
    n_states, n_actions, d, anchor, lambda_l1, verbose,
):
    # Decision variables: T per (s, a) and a single w in R^d
    T_var = np.empty((n_states, n_actions), dtype=object)
    for s in range(n_states):
        for a in range(n_actions):
            T_var[s, a] = cp.Variable(n_states, nonneg=True)

    w = cp.Variable(d)

    # Weighted L2 fit on T (paper Eq 10)
    obj_terms = []
    for s in range(n_states):
        for a in range(n_actions):
            wt = N[s, a]
            if wt.sum() == 0:
                obj_terms.append(1e-6 * cp.sum_squares(T_var[s, a] - T_mle[s, a]))
            else:
                diff = T_var[s, a] - T_mle[s, a]
                obj_terms.append(cp.sum(cp.multiply(wt, cp.square(diff))))

    # L1 penalty on w (sparsity / identifiability)
    if lambda_l1 > 0:
        obj_terms.append(lambda_l1 * cp.norm1(w))

    objective = cp.Minimize(cp.sum(obj_terms))

    constraints = []

    # Simplex on T
    for s in range(n_states):
        for a in range(n_actions):
            constraints.append(cp.sum(T_var[s, a]) == 1)

    # Anchor (optional) — fix R(s_a, a_a) = value
    if anchor is not None:
        s_a, a_a, val = anchor
        constraints.append(Phi[s_a, a_a] @ w == val)

    # Eps-ball constraints (Eq 8 / Eq 9), at observed states only.
    for s in range(n_states):
        if not visited_s[s]:
            continue
        valid = np.where(visited_sa[s])[0]
        invalid = np.where(~visited_sa[s])[0]

        for a in valid:
            for a_bad in invalid:
                # Eq 8: (Phi[s,a] - Phi[s,a_bad]) @ w + gamma * (T_sa - T_bad) v_lin >= eps
                dPhi = Phi[s, a] - Phi[s, a_bad]
                dT = T_var[s, a] - T_var[s, a_bad]
                constraints.append(dPhi @ w + gamma * dT @ v_lin >= epsilon)

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, a2 = int(valid[i]), int(valid[j])
                dPhi = Phi[s, a] - Phi[s, a2]
                dT = T_var[s, a] - T_var[s, a2]
                expr = dPhi @ w + gamma * dT @ v_lin
                constraints.append(expr <= epsilon)
                constraints.append(expr >= -epsilon)

    prob = cp.Problem(objective, constraints)
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
                prob.solve(verbose=False, **solver_kwargs)
                status = str(prob.status)
                if status in ("optimal", "optimal_inaccurate"):
                    break
            except (cp.SolverError, Exception):
                status = "solver_error"
                continue

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, None, None, str(prob.status)

    # Extract T (project onto simplex as safety)
    T_hat = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            v = T_var[s, a].value
            if v is None:
                T_hat[s, a] = T_mle[s, a]
            else:
                v = np.maximum(v, 0.0)
                tot = v.sum()
                T_hat[s, a] = (v / tot) if tot > 1e-12 else T_mle[s, a].copy()

    w_hat = np.asarray(w.value).ravel() if w.value is not None \
        else np.zeros(d)

    return T_hat, w_hat, float(prob.value), str(prob.status)
