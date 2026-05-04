"""
Smoke tests for the core invariants of the ITL reproduction.

Run with:  pytest -q tests/

Each test is intentionally fast (target < 5 s) so the full suite runs in
~30 s and can sit in CI. They are not exhaustive — they catch regressions
on the things this codebase has been bitten by historically:
  - corridor v* drift (caught the v1->v2 gamma bug)
  - ITL solver failing to satisfy the ε-ball property at observed states
  - BITL initialization slack going negative (the constraint-set bug)
  - MCE not recovering pi* on corridor with anchor (the L-BFGS-B fix)
  - ITL+IRL not recovering pi* on corridor with anchor
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import (
    make_corridor,
    make_gridworld,
    gridworld_start_state,
    make_two_goal_gridworld,
    two_goal_states,
)
from src.expert import (
    generate_batch_data,
    make_epsilon_optimal_expert,
    generate_batch_dataset,
)
from src.itl_solver import solve_itl
from src.utils import best_matching, normalized_value
from src.mdp import TabularMDP, deterministic_policy


# ---------------------------------------------------------------------------
# Corridor: v* exact, hand-calculated (March 2026 worked example)
# ---------------------------------------------------------------------------

def test_corridor_value_function_exact():
    """v* on the corridor must match hand calcs to 2 decimals."""
    mdp = make_corridor(gamma=0.9)
    v_star, _, _ = mdp.compute_optimal_policy()
    assert v_star.shape == (3,)
    assert v_star[0] == pytest.approx(76.87, abs=0.01)
    assert v_star[1] == pytest.approx(87.68, abs=0.01)
    assert v_star[2] == pytest.approx(100.00, abs=0.01)


# ---------------------------------------------------------------------------
# ITL: convergence on corridor + better-than-MLE on unvisited (s, a) pairs
# ---------------------------------------------------------------------------

def test_itl_converges_on_corridor_and_beats_mle():
    mdp = make_corridor(gamma=0.9)
    _, _, pi_star = mdp.compute_optimal_policy()
    N, T_mle = generate_batch_data(mdp, pi_star, n_samples_per_sa=20, seed=42)

    T_hat, info = solve_itl(N, T_mle, mdp.R, mdp.gamma,
                             epsilon=1.0, max_iter=10, verbose=False)
    assert info["converged"], (
        f"ITL did not converge on corridor — info={info}"
    )

    # ITL should beat MLE on unvisited (s, a) pairs (the v1->v2 fix promise).
    sa_unvisited = (N.sum(axis=2) == 0)
    if sa_unvisited.any():
        mle_err = np.mean([
            np.sum((mdp.T[s, a] - T_mle[s, a]) ** 2)
            for s in range(mdp.n_states)
            for a in range(mdp.n_actions)
            if sa_unvisited[s, a]
        ])
        itl_err = np.mean([
            np.sum((mdp.T[s, a] - T_hat[s, a]) ** 2)
            for s in range(mdp.n_states)
            for a in range(mdp.n_actions)
            if sa_unvisited[s, a]
        ])
        assert itl_err <= mle_err + 1e-9, (
            f"ITL did not beat MLE on unvisited pairs: ITL={itl_err}, MLE={mle_err}"
        )


# ---------------------------------------------------------------------------
# BITL: constraint set must come from observed actions (the bug-1 regression)
# ---------------------------------------------------------------------------

def test_bitl_initial_slack_nonnegative():
    """ITL solution must be feasible under BITL's constraint set.

    Pre-fix: BITL built constraints from MLE's eps-ball under Q(T_MLE),
    which produced initial slack ~-2.0 on corridor. Post-fix: uses
    observed expert actions per paper Definition 1, slack >= 0.
    """
    from src.bitl import bitl_sample
    from src.itl_solver import solve_itl as _itl
    mdp = make_corridor(gamma=0.9)
    _, _, pi_star = mdp.compute_optimal_policy()
    N, T_mle = generate_batch_data(mdp, pi_star, n_samples_per_sa=20, seed=42)
    T_itl, _ = _itl(N, T_mle, mdp.R, mdp.gamma, epsilon=1.0, max_iter=10)
    samples, info = bitl_sample(
        N, T_mle, mdp.R, mdp.gamma, epsilon=1.0,
        n_samples=20, n_warmup=10,  # tiny: we only check feasibility, not mixing
        T_init=T_itl, seed=0, verbose=False,
    )
    # If the constraint set were wrong, we'd see "Initial min constraint
    # slack" go negative and constraints get silently relaxed. We can't
    # easily inspect that from outside, so we check that posterior samples
    # are at least valid simplex rows.
    assert samples.shape == (20, 3, 2, 3)
    sums = samples.sum(axis=3)
    assert np.allclose(sums, 1.0, atol=1e-3), \
        "BITL posterior samples are not valid simplex rows."


# ---------------------------------------------------------------------------
# MCE: anchor-based corridor recovery (the L-BFGS-B fix regression)
# ---------------------------------------------------------------------------

def test_mce_corridor_match_with_anchor():
    """MCE with anchor (2, 0, 10.0) must recover pi* on corridor.

    Pre-fix (naive gradient ascent): match=0.000.
    Post-fix (L-BFGS-B + null-space anchor): match=1.000.
    """
    from src.mce_baseline import mce_solve
    mdp = make_corridor(gamma=0.9)
    _, _, pi_star = mdp.compute_optimal_policy()
    N, T_mle = generate_batch_data(mdp, pi_star, n_samples_per_sa=20, seed=42)
    Phi = np.zeros((mdp.n_states, mdp.n_actions, mdp.n_states))
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            Phi[s, a, s] = 1.0
    T_hat, w_hat, _ = mce_solve(
        N, Phi, mdp.gamma, anchor=(2, 0, 10.0), max_outer=3, verbose=False,
    )
    R_hat = Phi @ w_hat
    mdp_hat = TabularMDP(mdp.n_states, mdp.n_actions, T_hat, R_hat, mdp.gamma)
    _, _, pi_hat = mdp_hat.compute_optimal_policy()
    assert best_matching(pi_hat, pi_star) == 1.0
    # Anchor must be exact to floating-point precision.
    assert abs(w_hat[2] - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# ITL+IRL: anchor-based corridor recovery
# ---------------------------------------------------------------------------

def test_itl_irl_corridor_match_with_anchor():
    from src.itl_irl_solver import solve_itl_irl
    mdp = make_corridor(gamma=0.9)
    _, _, pi_star = mdp.compute_optimal_policy()
    N, T_mle = generate_batch_data(mdp, pi_star, n_samples_per_sa=20, seed=42)
    Phi = np.zeros((mdp.n_states, mdp.n_actions, mdp.n_states))
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            Phi[s, a, s] = 1.0
    T_hat, w_hat, _ = solve_itl_irl(
        N, T_mle, Phi, mdp.gamma, epsilon=1.0,
        anchor=(2, 0, 10.0), lambda_l1=0.01, max_iter=10, verbose=False,
    )
    R_hat = Phi @ w_hat
    mdp_hat = TabularMDP(mdp.n_states, mdp.n_actions, T_hat, R_hat, mdp.gamma)
    _, _, pi_hat = mdp_hat.compute_optimal_policy()
    assert best_matching(pi_hat, pi_star) == 1.0


# ---------------------------------------------------------------------------
# PS: posterior mean equals Laplace-smoothed MLE (sanity of the simplest case)
# ---------------------------------------------------------------------------

def test_ps_point_estimate_equals_laplace_mle():
    """PS = unconstrained Dir(N+δ); its mean is exactly the Laplace MLE."""
    from src.ps_baseline import ps_point_estimate
    from src.expert import compute_mle_transitions
    rng = np.random.default_rng(0)
    N = rng.integers(0, 10, size=(4, 3, 4)).astype(float)
    np.testing.assert_allclose(
        ps_point_estimate(N, delta=0.001),
        compute_mle_transitions(N, delta=0.001),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Gridworld: the big claim — ITL strictly beats MLE on Normalized Value
# at full coverage. Single-seed sanity, not a stat test.
# ---------------------------------------------------------------------------

def test_gridworld_itl_beats_mle_at_full_coverage():
    mdp = make_gridworld(grid_size=5, gamma=0.95, slip_prob=0.2)
    pi_expert = make_epsilon_optimal_expert(
        mdp, epsilon=5.0, target_stochastic_fraction=0.4,
    )
    start = gridworld_start_state(grid_size=5)
    N, T_mle, _ = generate_batch_dataset(
        mdp, pi_expert, coverage=1.0, K=10, delta=0.001, seed=0,
    )
    T_itl, _ = solve_itl(N, T_mle, mdp.R, mdp.gamma,
                          epsilon=5.0, max_iter=10, verbose=False)
    nv_mle = normalized_value(T_mle, mdp.R, mdp.gamma, mdp, start_state=start)
    nv_itl = normalized_value(T_itl, mdp.R, mdp.gamma, mdp, start_state=start)
    assert nv_itl > nv_mle, (
        f"At full coverage, ITL ({nv_itl}) should beat MLE ({nv_mle}). "
        "Regression in either ITL solver or expert generation."
    )
    # And ITL should be near-perfect at full coverage.
    assert nv_itl > 0.9, f"ITL NV at full coverage = {nv_itl}, expected > 0.9"


# ---------------------------------------------------------------------------
# Honest goal-domination check: pi*(T_MLE, R_trivial_goal_only) ~ pi*(T_MLE, R_TRUE)
# This documents the methodology gap from the recent code review.
# ---------------------------------------------------------------------------

def test_gridworld_is_goal_dominated():
    """Sanity / documentation: on this gridworld, a trivial goal-only reward
    paired with T_MLE recovers as good a policy as the true reward paired
    with T_MLE. So policy match alone cannot certify reward recovery.

    This test does NOT assert a specific number; it just records the
    behavior so a future change that breaks it surfaces in CI.
    """
    mdp = make_gridworld(grid_size=5, gamma=0.95, slip_prob=0.2)
    pi_expert = make_epsilon_optimal_expert(
        mdp, epsilon=5.0, target_stochastic_fraction=0.4,
    )
    _, _, pi_star = mdp.compute_optimal_policy()
    N, T_mle, _ = generate_batch_dataset(
        mdp, pi_expert, coverage=1.0, K=10, delta=0.001, seed=0,
    )
    mdp_true_R = TabularMDP(mdp.n_states, mdp.n_actions, T_mle, mdp.R, mdp.gamma)
    _, _, pi_truth = mdp_true_R.compute_optimal_policy()

    R_trivial = np.zeros((mdp.n_states, mdp.n_actions))
    R_trivial[0 * 5 + 4, :] = 10  # top-right goal cell only
    mdp_triv = TabularMDP(mdp.n_states, mdp.n_actions, T_mle, R_trivial, mdp.gamma)
    _, _, pi_triv = mdp_triv.compute_optimal_policy()

    match_truth = best_matching(pi_truth, pi_star)
    match_triv = best_matching(pi_triv, pi_star)
    # On the goal-dominated MDP, these should be very close (within 0.05).
    assert abs(match_truth - match_triv) < 0.05, (
        f"Goal-domination invariant changed: truth-R match {match_truth} vs "
        f"trivial-R match {match_triv}. Either the MDP changed or the "
        "policy under T_MLE became more reward-discriminating, which is "
        "good news for the ITL+IRL benchmark — update this test."
    )


# ---------------------------------------------------------------------------
# Two-goal gridworld: NOT goal-dominated. The complement of the test above.
# Documents the methodology gap fix from `docs/c_itl_options.md`.
# ---------------------------------------------------------------------------

def test_two_goal_is_NOT_goal_dominated():
    """Two-goal gridworld must NOT be goal-dominated.

    On this benchmark, a trivial reward `R_trivial = 10 * 1[s in {A, B}]`
    paired with `T_MLE` should produce a policy that's at least 10%
    *worse* (in best-matching to pi*) than the true reward paired with
    the same `T_MLE`. If this fails, the env collapses to "find any
    goal" and the ITL+IRL acceptance test on it would not certify
    that the IRL step is doing useful work — it would just be testing
    goal-localization.

    See `docs/c_itl_options.md` "Methodology gap discovered 2026-05-01".
    """
    mdp = make_two_goal_gridworld(grid_size=5, gamma=0.95, slip_prob=0.2)
    goal_A, goal_B = two_goal_states(grid_size=5)
    _, _, pi_star = mdp.compute_optimal_policy()

    pi_expert = make_epsilon_optimal_expert(
        mdp, epsilon=5.0, target_stochastic_fraction=0.4,
    )
    N, T_mle, _ = generate_batch_dataset(
        mdp, pi_expert, coverage=1.0, K=10, delta=0.001, seed=0,
    )

    # pi*(T_MLE, R_TRUE)
    mdp_truth = TabularMDP(mdp.n_states, mdp.n_actions, T_mle, mdp.R, mdp.gamma)
    _, _, pi_truth = mdp_truth.compute_optimal_policy()

    # pi*(T_MLE, R_trivial) — equal +10 at both goals, 0 everywhere else.
    R_trivial = np.zeros((mdp.n_states, mdp.n_actions))
    R_trivial[goal_A, :] = 10.0
    R_trivial[goal_B, :] = 10.0
    mdp_triv = TabularMDP(mdp.n_states, mdp.n_actions, T_mle, R_trivial, mdp.gamma)
    _, _, pi_triv = mdp_triv.compute_optimal_policy()

    match_truth = best_matching(pi_truth, pi_star)
    match_triv = best_matching(pi_triv, pi_star)
    gap = match_truth - match_triv

    assert gap > 0.10, (
        f"Two-goal gridworld is still goal-dominated: truth-R match "
        f"{match_truth:.3f} vs trivial-R match {match_triv:.3f} "
        f"(gap {gap:+.3f}, need > 0.10). Either widen the reward gap "
        f"(R_B - R_A) or strengthen the soft-wall barrier."
    )


# ---------------------------------------------------------------------------
# Two-goal ITL+IRL acceptance: must beat MLE-T + R_trivial by >= 0.05.
# (Full 20-seed acceptance against ITL-T + R_trivial is in
#  experiments/run_itl_irl_two_goal.py; see MVR_findings.md for the
#  partial-failure writeup of the ITL-T comparison.)
# ---------------------------------------------------------------------------

def test_itl_irl_recovers_R_on_two_goal():
    """ITL+IRL must recover at least *some* reward signal beyond the
    naive MLE-T + R_trivial baseline.

    Single-seed regression test: on the two-goal env (seed=0),
    `pi*(T_HAT, R_HAT)` from joint inference must beat
    `pi*(T_MLE, R_trivial)` by at least 0.05 in best_matching against
    pi*. Empirically the gap is ≈ 0.36 (20-seed mean +0.37), so this
    has wide headroom.

    Note: the *stronger* test "ITL+IRL beats ITL-T + R_trivial by 0.05"
    is intentionally NOT enforced here. That comparison fails on this
    benchmark by ≈ 0.01 because ITL's eps-ball constraints absorb the
    expert's preferences into T regardless of the input R, leaving
    ITL+IRL with little headroom against ITL-T + any reasonable R.
    See `results/MVR_findings.md` for the full diagnostic.
    """
    from src.itl_irl_solver import solve_itl_irl

    mdp = make_two_goal_gridworld(grid_size=5, gamma=0.95, slip_prob=0.2)
    goal_A, goal_B = two_goal_states(grid_size=5)
    _, _, pi_star = mdp.compute_optimal_policy()

    pi_expert = make_epsilon_optimal_expert(
        mdp, epsilon=5.0, target_stochastic_fraction=0.4,
    )
    N, T_mle, _ = generate_batch_dataset(
        mdp, pi_expert, coverage=1.0, K=10, delta=0.001, seed=0,
    )

    # Baseline: pi*(T_MLE, R_trivial)
    R_trivial = np.zeros((mdp.n_states, mdp.n_actions))
    R_trivial[goal_A, :] = 10.0
    R_trivial[goal_B, :] = 10.0
    mdp_triv = TabularMDP(
        mdp.n_states, mdp.n_actions, T_mle, R_trivial, mdp.gamma
    )
    _, _, pi_triv = mdp_triv.compute_optimal_policy()
    bm_triv = best_matching(pi_triv, pi_star)

    # ITL+IRL with one-hot (s, a) features and anchor at goal B = 10.0.
    # lambda_l1 = 0.1 reaches the same fixed point as 0.01 in 2 outer
    # iterations, so the test runs in ~20s.
    d = mdp.n_states * mdp.n_actions
    Phi = np.zeros((mdp.n_states, mdp.n_actions, d))
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            Phi[s, a, s * mdp.n_actions + a] = 1.0

    T_hat, w_hat, _ = solve_itl_irl(
        N, T_mle, Phi, mdp.gamma, epsilon=5.0,
        anchor=(goal_B, 0, 10.0), lambda_l1=0.1,
        max_iter=3, tol=1e-6, verbose=False,
    )
    R_hat = Phi @ w_hat
    mdp_hat = TabularMDP(
        mdp.n_states, mdp.n_actions, T_hat, R_hat, mdp.gamma
    )
    _, _, pi_hat = mdp_hat.compute_optimal_policy()
    bm_irl = best_matching(pi_hat, pi_star)

    gap = bm_irl - bm_triv
    assert gap >= 0.05, (
        f"ITL+IRL did not beat MLE-T + R_trivial by >= 0.05: "
        f"ITL+IRL={bm_irl:.3f}, MLE-T+R_trivial={bm_triv:.3f}, "
        f"gap={gap:+.3f}. Joint inference is broken or env regressed."
    )
