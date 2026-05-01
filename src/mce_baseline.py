"""
MCE = Maximum Causal Entropy IRL with simultaneous T+R estimation.

Reference: Herman, Krause, Doshi-Velez et al. 2016, "Inverse Reinforcement
Learning with Simultaneous Estimation of Rewards and Dynamics", AISTATS
2016. (We don't have the paper PDF in this session; the formulation below
is the standard MaxCausalEnt-IRL with a likelihood-based T-step, which
matches the published interface but may differ in some appendix details.
Verify before publication.)

Why MCE matters for the reproduction: the Benac et al. 2024 ITL paper
reports MCE as a baseline column in every results table (Table 4–8). It
is the strongest competitor that, like ITL+IRL, infers BOTH dynamics and
rewards from expert demonstrations.

Algorithm sketch:
  1. Initialize T = T_MLE, w = 0.
  2. R-step (MaxCausalEnt-IRL): for fixed T, find reward weights w that
     maximize the log-likelihood of the observed expert behavior under
     the soft-Bellman policy
        pi_softBellman(a | s) ∝ exp((Q^soft(s, a)) / temperature)
     where Q^soft is the soft-max over future states under T and R = Phi @ w.
  3. T-step: for fixed R, fit T to maximize the data likelihood
     log P(D | T) = sum_{s,a,s'} N[s,a,s'] log T[s'|s,a].
     With Dirichlet prior, this is the Laplace-smoothed MLE — same as
     T_MLE. So in the simplest form, the T-step is just T <- T_MLE.
     (More elaborate Herman-et-al-style versions would constrain T using
     the inferred reward; a placeholder for that is in the TODO below.)
  4. Iterate (2)–(3) until convergence (or single-shot R-step at fixed
     T = T_MLE, which is plain MaxEnt-IRL).

This file ships:
  - `mce_solve(...)`: full alternating loop with the simple T-step.
  - `maxent_irl_step(...)`: the R-step alone (for ablation / debugging).
  - `mce_point_estimate(...)`: returns (T_hat, R_hat) usable as a baseline
    column alongside ITL / MLE / PS.

STATUS (2026-05-01): R-step ported from naive gradient ascent to
`scipy.optimize.minimize` with L-BFGS-B. Anchor support added to break
the additive-constant ambiguity (Phi[s, a] @ w == value), implemented
by null-space parameterization w = w_p + N_basis @ z so L-BFGS-B
remains unconstrained and the constraint holds exactly.

Soft-Bellman backup + soft-policy machinery is unchanged (still
correct: with true w, it recovers pi* on the corridor). On the corridor
with anchor=(2, 0, 10.0), `mce_solve` recovers pi* with
best-action match = 1.0.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict

from scipy.optimize import minimize

from .mdp import TabularMDP


# ---------------------------------------------------------------------------
# Soft-Bellman backup (the policy form MaxCausalEnt assumes the expert uses)
# ---------------------------------------------------------------------------

def soft_bellman_q(
    T: np.ndarray, R: np.ndarray, gamma: float,
    temperature: float = 1.0, n_iter: int = 200, tol: float = 1e-7,
) -> np.ndarray:
    """
    Soft-Bellman backup for an entropy-regularized policy:
      V_soft(s)  = temperature * log sum_a exp(Q_soft(s, a) / temperature)
      Q_soft(s, a) = R(s, a) + gamma * sum_{s'} T(s'|s, a) V_soft(s')

    Cross-reference: Krause & Hubotter Ch 10 (entropy-regularized RL,
    soft-max Bellman).
    """
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    for _ in range(n_iter):
        expected_next = np.einsum("sab,b->sa", T, V)
        Q = R + gamma * expected_next
        # log-sum-exp over actions for V_soft
        m = Q.max(axis=1, keepdims=True)
        V_new = (m.squeeze(axis=1)
                 + temperature * np.log(np.exp((Q - m) / temperature).sum(axis=1)))
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    expected_next = np.einsum("sab,b->sa", T, V)
    Q = R + gamma * expected_next
    return Q


def soft_policy(Q: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Soft-max policy from soft Q-values. Numerically stable via shift."""
    Q_shifted = Q / temperature
    m = Q_shifted.max(axis=1, keepdims=True)
    p = np.exp(Q_shifted - m)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# R-step: MaxCausalEnt IRL with fixed T
# ---------------------------------------------------------------------------

def _anchor_basis(
    Phi: np.ndarray, anchor: Tuple[int, int, float]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build the null-space parameterization for an anchor Phi[s, a] @ w = value.

    Returns (w_p, N_basis, nrm_sq) such that:
      - w_p satisfies Phi[s, a] @ w_p = value (particular solution).
      - N_basis (shape (d, d-1)) spans ker(Phi[s, a]).
      - Any feasible w can be written w = w_p + N_basis @ z, z in R^{d-1}.

    nrm_sq = ||Phi[s, a]||^2 is returned for projection / debugging.
    Raises ValueError if Phi[s, a] is the zero vector (no anchor possible).
    """
    s_a, a_a, val = anchor
    phi_anchor = Phi[s_a, a_a].astype(float)
    nrm_sq = float(phi_anchor @ phi_anchor)
    if nrm_sq < 1e-12:
        raise ValueError(
            f"Anchor at (s={s_a}, a={a_a}) has zero feature vector; "
            f"cannot enforce Phi[s, a] @ w = {val}."
        )
    w_p = (val / nrm_sq) * phi_anchor
    # Right-singular vectors of phi_anchor[None, :] orthogonal to phi_anchor
    # span ker(phi_anchor). With a 1-by-d row, the kernel has dim d - 1.
    _, _, Vh = np.linalg.svd(phi_anchor[None, :], full_matrices=True)
    N_basis = Vh.T[:, 1:]  # (d, d-1)
    return w_p, N_basis, nrm_sq


def maxent_irl_step(
    N: np.ndarray, T: np.ndarray, Phi: np.ndarray, gamma: float,
    w_init: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    anchor: Optional[Tuple[int, int, float]] = None,
    n_iter: int = 200,
    tol: float = 1e-7,
    verbose: bool = False,
    # legacy params (ignored; kept so old call sites don't break)
    lr: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Maximum Causal Entropy IRL R-step. Find w that minimizes the negative
    log-likelihood of expert demos under the soft-Bellman policy:

        -log L(w) = -sum_{s,a} N[s,a] * log pi_softBellman(a | s; w)

    Solved via scipy.optimize.minimize with L-BFGS-B. The gradient passed
    to L-BFGS-B is the standard MaxCausalEnt moment-matching gradient
    (sign-flipped for minimization):

        d (-log L) / dw  =  expected_feature_count - expert_feature_count

    where expected_feature_count is the discounted (s, a) visitation under
    pi_softBellman, computed via `_stationary_state_dist`.

    Args:
        N: (S, A, S) transition counts.
        T: (S, A, S) fixed transition matrix (T-step has already run).
        Phi: (S, A, d) reward features. R(s, a) = Phi[s, a] @ w.
        gamma: discount factor.
        w_init: optional starting point (default zeros, projected onto the
            anchor constraint set if `anchor` is provided).
        temperature: soft-Bellman temperature.
        anchor: optional (s, a, value) breaking the additive-constant
            ambiguity by enforcing Phi[s, a] @ w = value exactly.
            Implemented by null-space parameterization w = w_p + N_basis @ z;
            L-BFGS-B then optimizes z in R^{d-1} unconstrained.
        n_iter: max L-BFGS-B iterations.
        tol: L-BFGS-B gradient tolerance.
        verbose: print convergence summary.
        lr: ignored (legacy from gradient-ascent version).

    Returns:
        (w_hat, info)
    """
    del lr  # legacy, unused
    n_states, n_actions, d = Phi.shape

    # Empirical (s, a) visitation, normalized so it integrates to 1 (matches
    # the model_phi normalization, which uses d_state @ pi summing to 1).
    sa_visits = N.sum(axis=2)                  # (S, A)
    total_visits = float(sa_visits.sum())
    if total_visits == 0:
        return (np.zeros(d) if w_init is None else w_init.copy()), {
            "converged": False, "n_iter": 0, "reason": "no_data",
        }
    expert_phi = np.einsum("sa,sad->d", sa_visits, Phi) / total_visits

    # ----- Null-space parameterization for the anchor (or identity if none) -----
    if anchor is not None:
        w_p, N_basis, nrm_sq = _anchor_basis(Phi, anchor)
    else:
        w_p = np.zeros(d)
        N_basis = np.eye(d)
        nrm_sq = 1.0  # unused

    # ----- Initial w and its z-coordinate -----
    if w_init is None:
        w0 = w_p.copy()
    else:
        w0 = w_init.astype(float).copy()
        if anchor is not None:
            s_a, a_a, val = anchor
            phi_anchor = Phi[s_a, a_a].astype(float)
            residual = float(phi_anchor @ w0 - val)
            w0 = w0 - (residual / nrm_sq) * phi_anchor
    # w0 = w_p + N_basis @ z0  =>  z0 = N_basis.T @ (w0 - w_p)
    z0 = N_basis.T @ (w0 - w_p)

    # ----- Objective and gradient over z (unconstrained) -----
    # We use the EXACT analytic gradient of the data NLL under fixed T.
    # The "standard MaxCausalEnt gradient" expert_phi - expected_phi is the
    # moment-matching dual gradient and is *not* the gradient of the data
    # NLL — passing both to L-BFGS-B confuses line search. Below we derive
    # d(NLL)/dw via the soft-Bellman chain rule:
    #   log pi_soft(a|s) = Q_soft(s,a)/T - V_soft(s)/T
    #   dlog pi_soft(a|s)/dw = (1/T) * (G(s,a) - sum_b pi_soft(b|s) G(s,b))
    #   where G(s,a) = dQ_soft(s,a)/dw = Phi(s,a) + gamma * sum_{s'} T(s'|s,a) H(s')
    #   and H(s) = dV_soft(s)/dw = sum_a pi_soft(a|s) G(s,a).
    # Eliminating G yields the policy-evaluation linear system
    #   H = phi_pi + gamma * P_pi @ H
    # so H = (I - gamma P_pi)^{-1} phi_pi (S × d).
    # Then d(NLL)/dw = -(1/T) * sum_{s,a} m(s,a) * G(s,a),
    # where m(s,a) = N[s,a] - N(s) * pi_soft(a|s) is the visitation residual.
    def objective_and_grad(z: np.ndarray) -> Tuple[float, np.ndarray]:
        w = w_p + N_basis @ z
        R = Phi @ w                                         # (S, A)
        Q = soft_bellman_q(T, R, gamma, temperature=temperature)

        Q_t = Q / temperature
        mx = Q_t.max(axis=1, keepdims=True)
        log_Z = mx.squeeze(axis=1) + np.log(np.exp(Q_t - mx).sum(axis=1))
        log_pi = Q_t - log_Z[:, None]                       # (S, A)
        pi = np.exp(log_pi)                                 # (S, A) — soft policy

        nll = -float(np.sum(sa_visits * log_pi)) / total_visits

        # Visitation residual at each (s, a): observed minus expected.
        N_s = sa_visits.sum(axis=1)                         # (S,)
        m_resid = sa_visits - N_s[:, None] * pi             # (S, A)

        # Solve H = (I - gamma * P_pi)^{-1} phi_pi  with phi_pi(s) = E_a[Phi(s,a)].
        P_pi = np.einsum("sa,sab->sb", pi, T)               # (S, S')
        phi_pi = np.einsum("sa,sad->sd", pi, Phi)           # (S, d)
        A_mat = np.eye(n_states) - gamma * P_pi
        H = np.linalg.solve(A_mat, phi_pi)                  # (S, d)

        # G(s, a) = Phi(s, a) + gamma * sum_{s'} T(s'|s,a) H(s')
        # d(NLL)/dw = -(1/T) * (sum_{s,a} m_resid Phi + gamma * sum_{s,a,s'} m_resid T H)
        term1 = np.einsum("sa,sad->d", m_resid, Phi)
        TH = np.einsum("sab,bd->sad", T, H)                 # (S, A, d)
        term2 = gamma * np.einsum("sa,sad->d", m_resid, TH)
        grad_w = -(term1 + term2) / (temperature * total_visits)
        grad_z = N_basis.T @ grad_w
        return nll, grad_z

    res = minimize(
        objective_and_grad, z0,
        method="L-BFGS-B", jac=True,
        options={"maxiter": int(n_iter), "gtol": float(tol), "disp": False},
    )

    w_hat = w_p + N_basis @ res.x

    # Report anchor residual (must be ~0 by construction).
    if anchor is not None:
        s_a, a_a, val = anchor
        anchor_resid = float(Phi[s_a, a_a] @ w_hat - val)
    else:
        anchor_resid = 0.0

    info = {
        "converged": bool(res.success),
        "n_iter": int(res.nit),
        "fun": float(res.fun),
        "grad_norm_inf": float(np.max(np.abs(res.jac))) if res.jac is not None else 0.0,
        "message": str(res.message),
        "anchor_residual": anchor_resid,
    }
    if verbose:
        print(f"    [maxent_irl_step] L-BFGS-B: converged={info['converged']} "
              f"nit={info['n_iter']} nll={info['fun']:.6f} "
              f"|grad|_inf={info['grad_norm_inf']:.2e} "
              f"anchor_resid={anchor_resid:.2e}")

    return w_hat, info


def _stationary_state_dist(
    T: np.ndarray, pi: np.ndarray, gamma: float, n_iter: int = 200,
) -> np.ndarray:
    """
    Discounted state visitation under (T, pi). Solves
      d_pi(s) = (1 - gamma) * mu_0(s) + gamma * sum_{s', a} d_pi(s') pi(a|s') T(s|s', a)
    via fixed-point iteration with mu_0 uniform.
    """
    n_states = T.shape[0]
    mu_0 = np.ones(n_states) / n_states
    d = mu_0.copy()
    P_pi = np.einsum("sa,sab->sb", pi, T)      # (S, S')
    for _ in range(n_iter):
        d_new = (1 - gamma) * mu_0 + gamma * (d @ P_pi)
        if np.max(np.abs(d_new - d)) < 1e-9:
            d = d_new
            break
        d = d_new
    return d


# ---------------------------------------------------------------------------
# T-step: simple Laplace-smoothed MLE (placeholder for full Herman et al.)
# ---------------------------------------------------------------------------

def mle_t_step(N: np.ndarray, delta: float = 0.001) -> np.ndarray:
    """T-step: posterior mean of Dir(N + delta). Same as paper Eq 5."""
    smoothed = N + delta
    return smoothed / smoothed.sum(axis=2, keepdims=True)


# ---------------------------------------------------------------------------
# Full MCE: alternation
# ---------------------------------------------------------------------------

def mce_solve(
    N: np.ndarray, Phi: np.ndarray, gamma: float,
    delta: float = 0.001,
    temperature: float = 1.0,
    max_outer: int = 5,
    irl_iter: int = 200,
    irl_tol: float = 1e-7,
    anchor: Optional[Tuple[int, int, float]] = None,
    w_init: Optional[np.ndarray] = None,
    verbose: bool = False,
    # legacy param kept for backwards-compat (ignored)
    irl_lr: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    MCE alternating solver. Returns (T_hat, w_hat, info).

    Iteration:
      T  <- mle_t_step(N) (in this simple form, T is fixed at T_MLE)
      w  <- maxent_irl_step(N, T, Phi, ..., anchor=anchor)
      Repeat. With the simple non-adaptive T-step a single outer iteration
      typically suffices; we keep the loop for forward-compat with a richer
      T-step.

    Args:
        N, Phi, gamma, delta, temperature: as in `maxent_irl_step`.
        max_outer: number of T/R alternations.
        irl_iter, irl_tol: forwarded to L-BFGS-B inside the R-step.
        anchor: optional (s, a, value) breaking additive-constant ambiguity.
        w_init: optional starting w (default: zeros, or the anchor's
            particular solution if `anchor` is set).
    """
    del irl_lr  # legacy, unused
    info: Dict = {"outer_iters": []}
    T_hat = mle_t_step(N, delta=delta)

    if w_init is None:
        w_hat = np.zeros(Phi.shape[2])
    else:
        w_hat = w_init.astype(float).copy()

    for it in range(max_outer):
        w_new, ri = maxent_irl_step(
            N, T_hat, Phi, gamma,
            w_init=w_hat,
            temperature=temperature,
            anchor=anchor,
            n_iter=irl_iter, tol=irl_tol,
            verbose=verbose,
        )
        change = float(np.max(np.abs(w_new - w_hat)))
        info["outer_iters"].append({
            "it": it, "w_change": change,
            "irl_converged": ri.get("converged", False),
            "irl_nit": ri.get("n_iter", 0),
            "anchor_residual": ri.get("anchor_residual", 0.0),
        })
        if verbose:
            print(f"  outer {it}: |dw|={change:.6f} "
                  f"irl_nit={ri.get('n_iter', 0)} "
                  f"irl_converged={ri.get('converged', False)}")
        w_hat = w_new
        if change < 1e-6:
            info["termination"] = "fixed_point"
            break
    else:
        info["termination"] = "max_outer"

    # NOTE: The published Herman et al. 2016 method has a more elaborate
    # T-step that constrains T using the inferred reward (the dynamics must
    # be consistent with the soft-Bellman policy that explains the demos).
    # Implementing that requires a non-convex projection step; we leave it
    # as TODO. For most downstream metrics (best matching, eps-matching,
    # CVaR), the simpler form here is a reasonable lower bound on what
    # the full Herman et al. method would deliver.

    return T_hat, w_hat, info


def mce_point_estimate(
    N: np.ndarray, Phi: np.ndarray, gamma: float,
    delta: float = 0.001, temperature: float = 1.0,
    anchor: Optional[Tuple[int, int, float]] = None,
    w_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: returns (T_hat, R_hat) for plugging into
    experiment scripts as a baseline column alongside ITL / MLE / PS.

    Pass `anchor=(s, a, value)` to break the reward additive-constant
    ambiguity (Phi[s, a] @ w = value).
    """
    T_hat, w_hat, _ = mce_solve(
        N, Phi, gamma,
        delta=delta, temperature=temperature,
        anchor=anchor, w_init=w_init,
    )
    R_hat = Phi @ w_hat
    return T_hat, R_hat
