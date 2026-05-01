"""
Gridworld reproduction — Benac et al. (2024), Table 4 (Gridworld, 40% stochastic
expert, standard task) and Figure 2 (Normalized Value vs. Coverage).

Default (fast) run: computes MLE and ITL columns only.
Set RUN_BASELINES=1 to also compute PS and MCE columns (paper Table 4
baselines that BITL/ITL are supposed to beat).
Set RUN_BITL=1 to additionally compute BITL posterior-mean and Value CVaR.
BITL is slowest (HMC over the constrained simplex), so it's gated.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_gridworld, gridworld_start_state
from src.expert import make_epsilon_optimal_expert, generate_batch_dataset
from src.itl_solver import solve_itl
from src.utils import (
    best_matching,
    epsilon_matching,
    normalized_value,
    total_variation,
    count_constraint_violations,
    transition_mse,
    summarize_runs,
    plot_coverage_sensitivity,
    value_cvar_from_T_distribution,
    value_cvar_from_point_T,
)
from src.mdp import TabularMDP


def evaluate(T_hat, R, gamma, true_mdp, epsilon, start_state):
    """Compute all paper metrics for a single learned T_hat."""
    mdp_learned = TabularMDP(
        true_mdp.n_states, true_mdp.n_actions, T_hat, R, gamma
    )
    _, _, pi_hat = mdp_learned.compute_optimal_policy()
    _, Q_star, pi_star = true_mdp.compute_optimal_policy()

    return {
        "normalized_value": normalized_value(
            T_hat, R, gamma, true_mdp, start_state=start_state
        ),
        "best_matching": best_matching(pi_hat, pi_star),
        "epsilon_matching": epsilon_matching(pi_hat, Q_star, epsilon),
        "total_variation": total_variation(true_mdp.T, T_hat),
        "n_violations": count_constraint_violations(
            T_hat, R, gamma, true_mdp, epsilon
        ),
        "mse": transition_mse(true_mdp.T, T_hat),
    }


def _value_cvar_block(T_dist_samples, R, gamma, true_mdp, start_state, seed=None):
    """Compute Value CVaR at 1%, 2%, 5% from a (K, S, A, S) sample stack."""
    return {
        f"value_cvar_{int(a*100)}": value_cvar_from_T_distribution(
            T_dist_samples, R, gamma, true_mdp, alpha=a, start_state=start_state,
        )
        for a in (0.01, 0.02, 0.05)
    }


def run_one(mdp, pi_expert, coverage, epsilon, K, delta, seed):
    """Generate batch data and compute every method's T estimate.

    Returns a dict {method_name: T_hat or sample_stack} plus the data N.
    Methods marked "samples" return (K, S, A, S) posterior samples; the
    point-estimate methods return just (S, A, S).

    Gated by RUN_BASELINES (PS, MCE) and RUN_BITL env vars to keep the
    fast path fast.
    """
    N, T_mle, _ = generate_batch_dataset(
        mdp, pi_expert, coverage=coverage, K=K, delta=delta, seed=seed
    )
    T_itl, _ = solve_itl(
        N, T_mle, mdp.R, mdp.gamma, epsilon=epsilon, max_iter=10, verbose=False
    )

    out = {"mle_T": T_mle, "itl_T": T_itl}

    if os.environ.get("RUN_BASELINES", "0") == "1":
        # PS = unconstrained Dir-Categorical posterior. Fast.
        from src.ps_baseline import ps_sample, ps_point_estimate
        ps_samples, _ = ps_sample(N, delta=delta, n_samples=200, seed=seed)
        out["ps_T"] = ps_point_estimate(N, delta=delta)
        out["ps_samples"] = ps_samples

        # MCE = MaxCausalEnt with simultaneous T+R inference. Uses one-hot
        # state-action features so we don't impose a hand-designed
        # feature map at this stage.
        from src.mce_baseline import mce_solve
        n_states, n_actions = mdp.n_states, mdp.n_actions
        Phi = np.zeros((n_states, n_actions, n_states * n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                Phi[s, a, s * n_actions + a] = 1.0
        try:
            T_mce, w_mce, _ = mce_solve(N, Phi, mdp.gamma, delta=delta,
                                         max_outer=3, verbose=False)
            out["mce_T"] = T_mce
        except Exception as e:
            out["mce_T"] = None
            out["mce_error"] = str(e)

    if os.environ.get("RUN_BITL", "0") == "1":
        from src.bitl import bitl_sample
        try:
            samples, _ = bitl_sample(
                N, T_mle, mdp.R, mdp.gamma, epsilon=epsilon,
                n_samples=200, n_warmup=100,
                step_size=0.005, n_leapfrog=10, delta=delta,
                T_init=T_itl, seed=seed, verbose=False,
            )
            out["bitl_T"] = samples.mean(axis=0)
            out["bitl_samples"] = samples
        except Exception as e:
            out["bitl_T"] = None
            out["bitl_error"] = str(e)

    return N, out


def main():
    print("=" * 70)
    print("  GRIDWORLD — Benac et al. (2024) MVR (Table 4 / Figure 2)")
    print("=" * 70)

    # Paper constants
    GAMMA = 0.95
    DELTA = 0.001
    EPSILON = 5.0                 # main ε from paper
    # 40% / 20% / 0% stochastic-policy-states are paper's Tables 4 / 5 / 6.
    # Set STOCHASTIC_FRACTION env var to switch ablations.
    STOCHASTIC_FRACTION = float(os.environ.get("STOCHASTIC_FRACTION", "0.4"))
    K = 10                        # paper's Gridworld K
    COVERAGES = [0.2, 0.4, 0.6, 0.8, 1.0]
    N_SEEDS = int(os.environ.get("N_SEEDS", "10"))  # MVR default 10 (paper uses 50)

    mdp = make_gridworld(
        grid_size=5, gamma=GAMMA, slip_prob=0.2, transfer=False
    )
    print(f"  States: {mdp.n_states}, Actions: {mdp.n_actions}")
    v_star, Q_star, pi_star = mdp.compute_optimal_policy()
    print(f"  V* range: [{v_star.min():.2f}, {v_star.max():.2f}]")

    pi_expert = make_epsilon_optimal_expert(
        mdp,
        epsilon=EPSILON,
        target_stochastic_fraction=STOCHASTIC_FRACTION,
    )
    n_stochastic = int(((pi_expert > 0).sum(axis=1) > 1).sum())
    print(f"  Expert: {n_stochastic}/{mdp.n_states} stochastic-policy states "
          f"(target {STOCHASTIC_FRACTION:.0%})")

    start_state = gridworld_start_state(grid_size=5)
    print(f"  Start state: {start_state} (bottom-left)")

    # --- Coverage sweep with per-seed checkpointing ---
    # Checkpoint files are namespaced by stochastic fraction so 40%/20%/0%
    # ablations don't clobber each other. The legacy non-namespaced file
    # (gridworld_seeds.json) corresponds to 40% and is migrated transparently.
    os.makedirs("results/checkpoints", exist_ok=True)
    sf_tag = f"sf{int(round(STOCHASTIC_FRACTION * 100)):03d}"
    ckpt_path = f"results/checkpoints/gridworld_seeds_{sf_tag}.json"
    legacy_path = "results/checkpoints/gridworld_seeds.json"
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            checkpoint = json.load(f)
    elif STOCHASTIC_FRACTION == 0.4 and os.path.exists(legacy_path):
        # Adopt legacy 40% checkpoint without re-running.
        with open(legacy_path) as f:
            checkpoint = json.load(f)
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f)
    else:
        checkpoint = {}

    # Methods this run will compute. Always at least mle + itl. Extras are
    # opt-in via env var to keep the fast path fast.
    methods = ["mle", "itl"]
    if os.environ.get("RUN_BASELINES", "0") == "1":
        methods += ["ps", "mce"]
    if os.environ.get("RUN_BITL", "0") == "1":
        methods += ["bitl"]

    metric_keys = ["normalized_value", "best_matching", "epsilon_matching",
                    "total_variation", "n_violations", "mse"]

    rows = []
    for coverage in COVERAGES:
        per_method_runs = {m: {k: [] for k in metric_keys} for m in methods}

        for seed in range(N_SEEDS):
            key = f"{coverage}:{seed}"
            entry = checkpoint.get(key, {})

            # Re-use cached metrics for any methods already present, compute
            # the missing ones from scratch in a single run_one call.
            # An entry is "stale" if it lacks CVaR keys — those were added
            # later, so old checkpoints predate them. Treat stale = missing.
            def _stale(m):
                em = entry.get(m)
                return em is not None and "value_cvar_5" not in em
            missing_methods = [
                m for m in methods if (m not in entry) or _stale(m)
            ]
            if missing_methods:
                N, T_outputs = run_one(
                    mdp, pi_expert, coverage, EPSILON, K, DELTA, seed=seed
                )
                for m in missing_methods:
                    T_hat = T_outputs.get(f"{m}_T")
                    if T_hat is None:
                        # Method failed (e.g., MCE solver error).
                        entry[m] = None
                        continue
                    metrics = evaluate(T_hat, mdp.R, mdp.gamma, mdp, EPSILON, start_state)
                    # Add Value CVaR. Use posterior samples when available
                    # (BITL, PS), bootstrap from Dir(N + delta) otherwise.
                    sample_key = f"{m}_samples"
                    if sample_key in T_outputs:
                        cvar = _value_cvar_block(
                            T_outputs[sample_key], mdp.R, mdp.gamma, mdp,
                            start_state,
                        )
                    else:
                        cvar = {
                            f"value_cvar_{int(a*100)}": value_cvar_from_point_T(
                                T_hat, N, mdp.R, mdp.gamma, mdp,
                                alpha=a, start_state=start_state,
                                n_bootstrap=100, delta=DELTA, seed=seed,
                            )
                            for a in (0.01, 0.02, 0.05)
                        }
                    metrics.update(cvar)
                    entry[m] = metrics

                checkpoint[key] = entry
                with open(ckpt_path, "w") as f:
                    json.dump(checkpoint, f)

            for m in methods:
                metrics = entry.get(m)
                if metrics is None:
                    continue
                for k in metric_keys:
                    per_method_runs[m][k].append(metrics[k])

        row = {"coverage": coverage}
        for m in methods:
            for k in metric_keys:
                if per_method_runs[m][k]:
                    mean_, std_ = summarize_runs(per_method_runs[m][k])
                    row[f"{m}_{k}_mean"] = mean_
                    row[f"{m}_{k}_std"] = std_
        rows.append(row)

        # Console summary: ITL vs MLE always; baselines if present.
        print(f"\n  Coverage = {coverage:.1f}  (avg over {N_SEEDS} seeds)")
        for m in methods:
            nv_mean = row.get(f"{m}_normalized_value_mean")
            nv_std = row.get(f"{m}_normalized_value_std")
            bm_mean = row.get(f"{m}_best_matching_mean")
            bm_std = row.get(f"{m}_best_matching_std")
            if nv_mean is None:
                continue
            print(f"    {m.upper():<5s}  NV: {nv_mean:+.3f} ± {nv_std:.3f}   "
                  f"BM: {bm_mean:.3f} ± {bm_std:.3f}   "
                  f"viol: {row.get(f'{m}_n_violations_mean', 0):.2f}")

    # Save (output table also namespaced by stochastic fraction).
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    out_path = f"results/tables/gridworld_coverage_sweep_{sf_tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "gamma": GAMMA, "delta": DELTA, "epsilon": EPSILON,
                "stochastic_fraction": STOCHASTIC_FRACTION,
                "K": K, "n_seeds": N_SEEDS, "coverages": COVERAGES,
            },
            "rows": rows,
        }, f, indent=2)

    sf_pct = int(round(STOCHASTIC_FRACTION * 100))
    plot_coverage_sensitivity(
        coverages=[r["coverage"] for r in rows],
        itl_mean=[r["itl_normalized_value_mean"] for r in rows],
        itl_std=[r["itl_normalized_value_std"] for r in rows],
        mle_mean=[r["mle_normalized_value_mean"] for r in rows],
        mle_std=[r["mle_normalized_value_std"] for r in rows],
        metric_name="Normalized Value",
        title=f"Gridworld — ITL vs MLE (ε={EPSILON}, {sf_pct}% stochastic expert)",
        save_path=f"results/figures/gridworld_normalized_value_vs_coverage_{sf_tag}.png",
    )
    plot_coverage_sensitivity(
        coverages=[r["coverage"] for r in rows],
        itl_mean=[r["itl_best_matching_mean"] for r in rows],
        itl_std=[r["itl_best_matching_std"] for r in rows],
        mle_mean=[r["mle_best_matching_mean"] for r in rows],
        mle_std=[r["mle_best_matching_std"] for r in rows],
        metric_name="Best matching",
        title=f"Gridworld — Best matching vs Coverage ({sf_pct}% stochastic)",
        save_path=f"results/figures/gridworld_best_matching_vs_coverage_{sf_tag}.png",
    )

    print(f"\n  Results written to {out_path}")
    print("  Figures in results/figures/")


if __name__ == "__main__":
    main()
