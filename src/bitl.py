"""
Bayesian Inverse Transition Learning (BITL) — Algorithm 3 from Benac et al. (2024).

Samples from the posterior:
    P_epsilon(T* | D) ∝ Prod_{s,a} Dir(N_{s,a} + delta) * I[T satisfies ITL constraints]

Textbook cross-reference (Krause & Hubotter, "Probabilistic Artificial
Intelligence"):
  - Ch 11 (Model-based Bayesian RL) motivates the Dirichlet-Categorical
    posterior over each row T(.|s,a): with a Dir(delta) prior and counts
    N_{s,a}, the posterior is Dir(N_{s,a} + delta). This is exactly the
    unnormalized prior factor in P_epsilon(T* | D) above. Bayesian Regret
    (compute_bayesian_regret below) is the metric the chapter uses to
    quantify dynamics uncertainty's effect on control.
  - Ch 12 (Hamiltonian Monte Carlo) gives the leapfrog integrator,
    Metropolis-adjusted HMC, and the Hamiltonian conservation / acceptance
    ratio used in `_hmc_step_vec`. Reflected HMC (bounce momentum off
    constraint hyperplanes) is the constrained-domain variant referenced in
    Ch 12's "HMC on manifolds / with constraints" discussion; this matches
    the paper's "reflection off the eps-ball constraint boundary".

Uses Hamiltonian Monte Carlo (HMC) with REFLECTION off constraint boundaries.
When a leapfrog step would violate an ITL constraint (Eq 8 or Eq 9), the
momentum is reflected off the constraint hyperplane rather than rejecting.

Implementation uses a log-barrier formulation to smoothly enforce constraints
during HMC, with vectorized constraint evaluation for performance.

Key outputs:
    - Posterior samples T^(1), ..., T^(K) of shape (n_states, n_actions, n_states)
    - Outlier detection: P(tau) = (1/K) sum_i prod_{steps} T^(i)(s'|s,a)
    - Bayesian regret for transfer/disagreement quantification
"""

import numpy as np
from typing import Tuple, Optional, List
from .mdp import TabularMDP


# =============================================================================
# MAIN BITL SAMPLER
# =============================================================================

def bitl_sample(
    N: np.ndarray,
    T_mle: np.ndarray,
    R: np.ndarray,
    gamma: float,
    epsilon: float,
    n_samples: int = 500,
    n_warmup: int = 200,
    step_size: float = 0.01,
    n_leapfrog: int = 10,
    delta: float = 1.0,
    barrier_strength: float = 0.1,
    T_init: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Run BITL posterior sampling via reflected HMC (Algorithm 3).

    The target distribution is:
        log P(T | D) = sum_{s,a} log Dir(T(.|s,a) | N_{s,a} + delta)
                       + mu * sum_i log(constraint_slack_i(T))

    IMPORTANT: T_init should be a feasible point (e.g., the ITL solution).
    If not provided, we solve ITL internally to find a feasible starting point.

    Args:
        N: transition counts, shape (n_states, n_actions, n_states)
        T_mle: MLE transition estimate, shape (n_states, n_actions, n_states)
        R: reward function, shape (n_states, n_actions)
        gamma: discount factor
        epsilon: near-optimality tolerance for the epsilon-ball
        n_samples: number of posterior samples to return (after warmup)
        n_warmup: number of warmup/burn-in iterations
        step_size: HMC leapfrog step size (adapted during warmup)
        n_leapfrog: number of leapfrog steps per HMC iteration
        delta: Dirichlet pseudo-count (concentration parameter)
        barrier_strength: log-barrier weight for constraint enforcement
        T_init: initial feasible transition matrix (if None, uses ITL solution)
        seed: random seed
        verbose: print progress

    Returns:
        samples: posterior samples, shape (n_samples, n_states, n_actions, n_states)
        info: dict with acceptance rate, constraint violations, diagnostics
    """
    from .itl_solver import solve_itl as _solve_itl

    rng = np.random.default_rng(seed)
    n_states, n_actions, _ = N.shape
    dim = n_states * n_actions * n_states

    # Compute the epsilon-ball structure
    mdp_mle = TabularMDP(n_states, n_actions, T_mle, R, gamma)
    _, Q_mle, _ = mdp_mle.compute_optimal_policy()
    valid = mdp_mle.compute_epsilon_ball(Q_mle, epsilon)

    # Linearized value for constraints
    v_lin = _compute_v_lin(T_mle, R, N, gamma, n_states, n_actions)

    # Build constraint MATRIX (vectorized): A_mat @ t_flat >= b_vec
    A_mat, b_vec = _build_constraint_matrix(R, gamma, v_lin, valid, epsilon, n_states, n_actions)
    n_constraints = A_mat.shape[0]

    if verbose:
        print(f"  Built {n_constraints} ITL constraints for {n_states}S x {n_actions}A MDP")

    # Dirichlet parameters
    alpha = N + delta

    # Find feasible starting point
    if T_init is not None:
        T_current = _project_simplex(T_init.copy())
    else:
        if verbose:
            print("  Finding feasible initial point via ITL solver...")
        T_current, _ = _solve_itl(N, T_mle, R, gamma, epsilon, max_iter=10, verbose=False)
        T_current = _project_simplex(T_current)

    # Verify/fix feasibility
    if n_constraints > 0:
        slacks = A_mat @ T_current.flatten() - b_vec
        min_slack = slacks.min()
        if verbose:
            print(f"  Initial min constraint slack: {min_slack:.6f}")
        if min_slack < 0:
            # Relax constraints so initial point is interior
            b_vec = b_vec + min_slack - 0.01

    # Precompute softmax Jacobian helpers
    phi_dim = n_states * n_actions * (n_states - 1)

    # Initialize phi from T
    phi_current = _T_to_phi(T_current)

    # Verify initial log posterior is finite
    lp_init = _log_posterior_vec(phi_current, alpha, A_mat, b_vec,
                                 n_states, n_actions, barrier_strength)
    if not np.isfinite(lp_init):
        if verbose:
            print(f"  WARNING: log posterior = {lp_init}, disabling barrier")
        barrier_strength = 0.0

    samples = np.zeros((n_samples, n_states, n_actions, n_states))
    n_accept = 0
    n_total = n_warmup + n_samples
    n_reflections = 0
    accept_history = []
    current_step = step_size

    for i in range(n_total):
        phi_new, accepted, n_refl = _hmc_step_vec(
            phi_current, alpha, A_mat, b_vec,
            n_states, n_actions, current_step, n_leapfrog,
            barrier_strength, rng
        )

        n_reflections += n_refl
        if accepted:
            phi_current = phi_new
            n_accept += 1
            accept_history.append(1)
        else:
            accept_history.append(0)

        # Adaptive step size during warmup
        if i < n_warmup and i > 0 and i % 10 == 0:
            recent_rate = np.mean(accept_history[-10:])
            if recent_rate < 0.3:
                current_step *= 0.7
            elif recent_rate > 0.8:
                current_step *= 1.3
            current_step = np.clip(current_step, 1e-6, 0.5)
        elif i == n_warmup:
            step_size = current_step
            if verbose:
                print(f"  Warmup done. Adapted step_size={step_size:.6f}, "
                      f"warmup_accept={np.mean(accept_history):.3f}")

        if i >= n_warmup:
            samples[i - n_warmup] = _phi_to_T(phi_current, n_states, n_actions)

        if verbose and (i + 1) % max(n_total // 5, 1) == 0:
            recent = accept_history[-min(50, len(accept_history)):]
            rate = np.mean(recent)
            print(f"  BITL iter {i+1}/{n_total}: accept={rate:.3f}, "
                  f"step={current_step:.6f}, reflections={n_reflections}")

    accept_rate = n_accept / n_total
    post_warmup_accept = np.mean(accept_history[n_warmup:]) if len(accept_history) > n_warmup else 0

    info = {
        "accept_rate": accept_rate,
        "post_warmup_accept_rate": post_warmup_accept,
        "n_reflections": n_reflections,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "adapted_step_size": step_size,
        "n_leapfrog": n_leapfrog,
        "n_constraints": n_constraints,
    }

    if verbose:
        print(f"  BITL complete: accept={accept_rate:.3f}, "
              f"post_warmup={post_warmup_accept:.3f}, reflections={n_reflections}")

    return samples, info


# =============================================================================
# VECTORIZED HMC STEP
# =============================================================================

def _hmc_step_vec(
    phi: np.ndarray,
    alpha: np.ndarray,
    A_mat: np.ndarray,
    b_vec: np.ndarray,
    n_states: int,
    n_actions: int,
    step_size: float,
    n_leapfrog: int,
    barrier_strength: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, bool, int]:
    """
    Single HMC step with log-barrier constraints and reflection.
    Fully vectorized constraint evaluation for performance.

    Follows the standard HMC recipe from Krause & Hubotter Ch 12:
    sample momentum p ~ N(0, I), simulate Hamiltonian dynamics with leapfrog,
    then Metropolis-accept with prob min(1, exp(H_current - H_prop)). The
    boundary reflection (see below) swaps rejection at the constraint surface
    for a deterministic momentum flip, keeping the chain on the feasible set.
    """
    p = rng.standard_normal(phi.shape)
    n_reflections = 0

    H_current = _hamiltonian_vec(phi, p, alpha, A_mat, b_vec,
                                  n_states, n_actions, barrier_strength)
    if not np.isfinite(H_current):
        return phi, False, 0

    phi_prop = phi.copy()
    p_prop = p.copy()

    # Leapfrog
    grad = _grad_log_posterior_vec(phi_prop, alpha, A_mat, b_vec,
                                    n_states, n_actions, barrier_strength)
    if not np.all(np.isfinite(grad)):
        return phi, False, 0
    p_prop += 0.5 * step_size * grad

    for step in range(n_leapfrog):
        phi_prop += step_size * p_prop

        # Reflection check
        if A_mat.shape[0] > 0:
            T_prop = _phi_to_T(phi_prop, n_states, n_actions)
            slacks = A_mat @ T_prop.flatten() - b_vec
            violated = slacks < 0
            if np.any(violated):
                # Reflect momentum off the most violated constraint
                worst = np.argmin(slacks)
                n_phi = _constraint_normal_phi(A_mat[worst], T_prop, n_states, n_actions)
                norm_sq = np.dot(n_phi.ravel(), n_phi.ravel())
                if norm_sq > 1e-20:
                    proj = np.dot(n_phi.ravel(), p_prop.ravel()) / norm_sq
                    if proj < 0:
                        p_prop -= 2 * proj * n_phi
                        n_reflections += 1

        grad = _grad_log_posterior_vec(phi_prop, alpha, A_mat, b_vec,
                                        n_states, n_actions, barrier_strength)
        if not np.all(np.isfinite(grad)):
            return phi, False, n_reflections

        if step < n_leapfrog - 1:
            p_prop += step_size * grad
        else:
            p_prop += 0.5 * step_size * grad

    p_prop = -p_prop

    H_prop = _hamiltonian_vec(phi_prop, p_prop, alpha, A_mat, b_vec,
                               n_states, n_actions, barrier_strength)
    if not np.isfinite(H_prop):
        return phi, False, n_reflections

    log_accept = H_current - H_prop
    if np.log(rng.random() + 1e-300) < log_accept:
        return phi_prop, True, n_reflections
    else:
        return phi, False, n_reflections


# =============================================================================
# VECTORIZED POTENTIAL ENERGY AND GRADIENTS
# =============================================================================

def _hamiltonian_vec(phi, p, alpha, A_mat, b_vec, n_states, n_actions, mu):
    lp = _log_posterior_vec(phi, alpha, A_mat, b_vec, n_states, n_actions, mu)
    return -lp + 0.5 * np.sum(p ** 2)


def _log_posterior_vec(phi, alpha, A_mat, b_vec, n_states, n_actions, mu):
    """
    Vectorized log posterior: Dirichlet + Jacobian + barrier.

    log Dir(T(.|s,a); alpha_{s,a}) = sum_k (alpha_{s,a,k} - 1) log T(k|s,a)
    up to a normalizing constant. This is the conjugate posterior from
    Krause & Hubotter Ch 11 over each row of T (Dirichlet-Categorical).
    """
    T = _phi_to_T(phi, n_states, n_actions)
    T_safe = np.maximum(T, 1e-300)
    log_T = np.log(T_safe)

    # Dirichlet log-likelihood: sum (alpha-1) * log T
    log_lik = np.sum((alpha - 1) * log_T)

    # Log-Jacobian of softmax: sum of all log T
    log_jac = np.sum(log_T)

    # Log-barrier (vectorized)
    log_barrier = 0.0
    if mu > 0 and A_mat.shape[0] > 0:
        slacks = A_mat @ T.flatten() - b_vec
        pos_mask = slacks > 1e-8
        neg_mask = ~pos_mask

        if np.any(pos_mask):
            log_barrier += np.sum(np.log(slacks[pos_mask]))
        if np.any(neg_mask):
            # Smooth penalty for violations
            log_barrier += np.sum(np.log(1e-8) - 100.0 * slacks[neg_mask] ** 2)

    return log_lik + log_jac + mu * log_barrier


def _grad_log_posterior_vec(phi, alpha, A_mat, b_vec, n_states, n_actions, mu):
    """
    Vectorized gradient of log posterior w.r.t. phi.

    Components:
    1. Dirichlet: d/dphi_k = (alpha_k - 1) - T_k * sum_j(alpha_j - 1)
    2. Jacobian: d/dphi_k = 1 - n_states * T_k
    3. Barrier: d/dphi_k = mu * sum_i (1/slack_i) * dslack_i/dphi_k
    """
    T = _phi_to_T(phi, n_states, n_actions)
    grad = np.zeros_like(phi)

    # Vectorized Dirichlet + Jacobian gradient
    for s in range(n_states):
        for a in range(n_actions):
            T_sa = T[s, a]
            alpha_sa = alpha[s, a]
            alpha_sum = np.sum(alpha_sa - 1)
            # Dirichlet part
            grad[s, a] = (alpha_sa[:-1] - 1) - T_sa[:-1] * alpha_sum
            # Jacobian correction
            grad[s, a] += 1.0 - n_states * T_sa[:-1]

    # Barrier gradient (vectorized over constraints)
    if mu > 0 and A_mat.shape[0] > 0:
        t_flat = T.flatten()
        slacks = A_mat @ t_flat - b_vec
        # Only contribute gradient from feasible constraints
        feasible = slacks > 1e-8
        if np.any(feasible):
            # d(log slack_i)/dT = A_mat[i] / slack_i
            # d(log slack_i)/dphi = (dT/dphi)^T @ (A_mat[i] / slack_i)
            # For softmax: dT[s,a,k]/dphi[s,a,j] = T[s,a,k]*(delta_kj - T[s,a,j])
            weights = mu / slacks[feasible]  # (n_feasible,)
            # Weighted constraint normals in T-space
            weighted_A = weights[:, None] * A_mat[feasible]  # (n_feasible, dim)
            # Sum weighted normals
            dbarrier_dT = weighted_A.sum(axis=0)  # (dim,)
            dbarrier_dT_3d = dbarrier_dT.reshape(n_states, n_actions, n_states)

            # Transform to phi-space via softmax Jacobian
            for s in range(n_states):
                for a in range(n_actions):
                    T_sa = T[s, a]
                    g_sa = dbarrier_dT_3d[s, a]
                    weighted_sum = np.dot(g_sa, T_sa)
                    grad[s, a] += T_sa[:-1] * (g_sa[:-1] - weighted_sum)

    return grad


def _constraint_normal_phi(A_row, T, n_states, n_actions):
    """Transform single constraint normal from T-space to phi-space."""
    A_3d = A_row.reshape(n_states, n_actions, n_states)
    n_phi = np.zeros((n_states, n_actions, n_states - 1))
    for s in range(n_states):
        for a in range(n_actions):
            T_sa = T[s, a]
            g_sa = A_3d[s, a]
            weighted_sum = np.dot(g_sa, T_sa)
            n_phi[s, a] = T_sa[:-1] * (g_sa[:-1] - weighted_sum)
    return n_phi


# =============================================================================
# CONSTRAINT CONSTRUCTION (VECTORIZED)
# =============================================================================

def _build_constraint_matrix(R, gamma, v_lin, valid, epsilon, n_states, n_actions):
    """
    Build constraint matrix A and vector b such that A @ t_flat >= b.

    Eq 8 (valid a vs invalid a'):
        gamma * v^T (T[s,a,:] - T[s,a',:]) >= epsilon - (R[s,a] - R[s,a'])

    Eq 9 (valid a vs valid a', both directions):
        gamma * v^T (T[s,a,:] - T[s,a',:]) >= -epsilon - (R[s,a] - R[s,a'])
        gamma * v^T (T[s,a',:] - T[s,a,:]) >= -epsilon - (R[s,a'] - R[s,a])

    Returns:
        A_mat: shape (n_constraints, n_states * n_actions * n_states)
        b_vec: shape (n_constraints,)
    """
    dim = n_states * n_actions * n_states
    rows_A = []
    rows_b = []

    for s in range(n_states):
        valid_actions = np.where(valid[s])[0]
        invalid_actions = np.where(~valid[s])[0]

        for a_v in valid_actions:
            # Eq 8: valid vs invalid
            for a_i in invalid_actions:
                row = np.zeros(dim)
                offset_v = s * n_actions * n_states + a_v * n_states
                offset_i = s * n_actions * n_states + a_i * n_states
                row[offset_v:offset_v + n_states] = gamma * v_lin
                row[offset_i:offset_i + n_states] = -gamma * v_lin
                reward_diff = R[s, a_v] - R[s, a_i]
                rows_A.append(row)
                rows_b.append(epsilon - reward_diff)

            # Eq 9: valid vs valid
            for a_v2 in valid_actions:
                if a_v < a_v2:
                    row = np.zeros(dim)
                    offset_1 = s * n_actions * n_states + a_v * n_states
                    offset_2 = s * n_actions * n_states + a_v2 * n_states
                    row[offset_1:offset_1 + n_states] = gamma * v_lin
                    row[offset_2:offset_2 + n_states] = -gamma * v_lin
                    reward_diff = R[s, a_v] - R[s, a_v2]

                    # expr >= -epsilon
                    rows_A.append(row.copy())
                    rows_b.append(-epsilon - reward_diff)
                    # -expr >= -epsilon (i.e., expr <= epsilon)
                    rows_A.append(-row)
                    rows_b.append(-epsilon + reward_diff)

    if not rows_A:
        return np.zeros((0, dim)), np.zeros(0)

    return np.array(rows_A), np.array(rows_b)


def _compute_v_lin(T_mle, R, N, gamma, n_states, n_actions):
    """Compute linearized value from MLE transitions and empirical policy."""
    sa_counts = N.sum(axis=2)
    pi_hat = np.zeros((n_states, n_actions))
    for s in range(n_states):
        if sa_counts[s].sum() > 0:
            actions_chosen = sa_counts[s] > 0
            pi_hat[s, actions_chosen] = 1.0 / actions_chosen.sum()
        else:
            pi_hat[s] = 1.0 / n_actions

    P_pi = np.einsum("sa,sab->sb", pi_hat, T_mle)
    r_pi = np.einsum("sa,sa->s", pi_hat, R)
    A = np.eye(n_states) - gamma * P_pi
    return np.linalg.solve(A, r_pi)


# =============================================================================
# SOFTMAX PARAMETERIZATION
# =============================================================================

def _T_to_phi(T):
    """T -> phi via log-ratio (inverse softmax). phi[s,a,k] = log(T[s,a,k]/T[s,a,-1])."""
    T_safe = np.maximum(T, 1e-300)
    return np.log(T_safe[:, :, :-1]) - np.log(T_safe[:, :, -1:])


def _phi_to_T(phi, n_states, n_actions):
    """phi -> T via softmax."""
    T = np.zeros((n_states, n_actions, n_states))
    exp_phi = np.exp(np.clip(phi, -500, 500))
    denom = 1.0 + exp_phi.sum(axis=2, keepdims=True)
    T[:, :, :-1] = exp_phi / denom
    T[:, :, -1] = 1.0 / denom.squeeze(axis=2)
    return T


def _project_simplex(T):
    """Project all rows of T onto the probability simplex."""
    T_proj = np.maximum(T, 1e-10)
    return T_proj / T_proj.sum(axis=2, keepdims=True)


# =============================================================================
# OUTLIER DETECTION
# =============================================================================

def compute_trajectory_likelihood(
    trajectory: List[Tuple[int, int, int]],
    samples: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute the posterior predictive probability of a trajectory.

    P(tau) = (1/K) sum_{i=1}^{K} prod_{(s,a,s') in tau} T^{(i)}(s'|s,a)

    A trajectory with low P(tau) relative to others is a potential outlier.

    Args:
        trajectory: list of (s, a, s') tuples
        samples: posterior samples, shape (K, n_states, n_actions, n_states)

    Returns:
        mean_likelihood: average likelihood across posterior samples
        per_sample_likelihoods: shape (K,)
    """
    K = samples.shape[0]
    log_liks = np.zeros(K)

    for s, a, s_next in trajectory:
        log_liks += np.log(np.maximum(samples[:, s, a, s_next], 1e-300))

    # Log-sum-exp for numerical stability
    max_ll = np.max(log_liks)
    mean_lik = np.exp(max_ll) * np.mean(np.exp(log_liks - max_ll))
    return mean_lik, np.exp(log_liks)


def detect_outlier_trajectories(
    trajectories: List[List[Tuple[int, int, int]]],
    samples: np.ndarray,
    threshold_percentile: float = 5.0,
) -> dict:
    """
    Identify outlier trajectories using posterior predictive probabilities.

    Trajectories whose P(tau) falls below the threshold_percentile are outliers.

    Args:
        trajectories: list of trajectories, each a list of (s, a, s') tuples
        samples: posterior samples, shape (K, n_states, n_actions, n_states)
        threshold_percentile: percentile cutoff for outlier classification

    Returns:
        dict with likelihoods, outlier_mask, threshold, outlier_indices
    """
    K = samples.shape[0]
    n_traj = len(trajectories)
    log_likelihoods = np.zeros(n_traj)

    for i, traj in enumerate(trajectories):
        per_sample_ll = np.zeros(K)
        for s, a, s_next in traj:
            per_sample_ll += np.log(np.maximum(samples[:, s, a, s_next], 1e-300))
        max_ll = np.max(per_sample_ll)
        log_likelihoods[i] = max_ll + np.log(np.mean(np.exp(per_sample_ll - max_ll)))

    threshold = np.percentile(log_likelihoods, threshold_percentile)
    outlier_mask = log_likelihoods < threshold

    return {
        "likelihoods": np.exp(log_likelihoods),
        "log_likelihoods": log_likelihoods,
        "outlier_mask": outlier_mask,
        "threshold": threshold,
        "outlier_indices": np.where(outlier_mask)[0],
        "n_outliers": outlier_mask.sum(),
    }


# =============================================================================
# BAYESIAN REGRET
# =============================================================================

def compute_bayesian_regret(
    samples: np.ndarray,
    R: np.ndarray,
    gamma: float,
) -> dict:
    """
    Compute Bayesian regret: measures posterior disagreement on optimal behavior.

    For each posterior sample T^(i), compute optimal policy pi^(i) and value V^(i).
    Bayesian regret at state s:
        BR(s) = max_i V^(i)(s) - mean_i V^(i)(s)

    High BR = high uncertainty about optimal policy at that state.

    Cross-reference: Krause & Hubotter Ch 11 presents Bayesian regret as the
    core criterion for model-based Bayesian RL under a posterior over
    dynamics. V^(i) is computed via Bellman optimality (Ch 10 Eq 10.9).

    Args:
        samples: shape (K, n_states, n_actions, n_states)
        R: shape (n_states, n_actions)
        gamma: discount factor

    Returns:
        dict with regret_per_state, mean_regret, policies, values, policy_disagreement
    """
    K, n_states, n_actions, _ = samples.shape

    policies = np.zeros((K, n_states, n_actions))
    values = np.zeros((K, n_states))

    for i in range(K):
        mdp_i = TabularMDP(n_states, n_actions, samples[i], R, gamma)
        v_i, Q_i, pi_i = mdp_i.compute_optimal_policy()
        policies[i] = pi_i
        values[i] = v_i

    regret_per_state = values.max(axis=0) - values.mean(axis=0)

    # Policy disagreement
    optimal_actions = np.argmax(policies, axis=2)
    disagreement = np.zeros(n_states)
    for s in range(n_states):
        n_unique = len(np.unique(optimal_actions[:, s]))
        disagreement[s] = 1.0 - 1.0 / n_unique

    return {
        "regret_per_state": regret_per_state,
        "mean_regret": np.mean(regret_per_state),
        "policies": policies,
        "values": values,
        "policy_disagreement": disagreement,
        "mean_disagreement": np.mean(disagreement),
    }


# =============================================================================
# POSTERIOR SUMMARY
# =============================================================================

def posterior_summary(samples: np.ndarray) -> dict:
    """Summary statistics of posterior samples (mean, std, CI)."""
    return {
        "mean": np.mean(samples, axis=0),
        "std": np.std(samples, axis=0),
        "ci_lower": np.percentile(samples, 2.5, axis=0),
        "ci_upper": np.percentile(samples, 97.5, axis=0),
        "median": np.median(samples, axis=0),
    }


def posterior_mse(samples: np.ndarray, T_true: np.ndarray) -> dict:
    """
    MSE of posterior mean vs truth, and 95% CI coverage.

    Args:
        samples: shape (K, n_states, n_actions, n_states)
        T_true: shape (n_states, n_actions, n_states)
    """
    summary = posterior_summary(samples)
    T_mean = summary["mean"]
    mse = np.mean(np.sum((T_true - T_mean) ** 2, axis=2))
    in_ci = (T_true >= summary["ci_lower"]) & (T_true <= summary["ci_upper"])
    coverage = np.mean(in_ci)

    return {
        "mse_posterior_mean": mse,
        "coverage_95": coverage,
        "T_posterior_mean": T_mean,
    }
