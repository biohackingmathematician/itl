"""
RandomWorld reproduction — Benac et al. (2024), Table 4 (Randomworlds, 40%
stochastic expert, standard task) and Figure 2 bottom row.

MVR: 5 worlds × 2 dataset seeds = 10 runs per coverage value.
Paper uses 20 × 5 = 100 runs (set N_WORLDS=20 N_DATASETS=5).

Default (fast) run: computes MLE and ITL columns only.
Set RUN_BASELINES=1 to also compute PS and MCE columns (paper Table 4
baselines that BITL/ITL are supposed to beat).
Set RUN_BITL=1 to additionally compute BITL posterior-mean.
BITL is slowest (HMC over the constrained simplex), so it's gated.

Existing checkpoints from the MLE+ITL-only era are reused as-is; only
methods that are actually missing from each checkpoint entry are recomputed.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments import make_randomworld
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
)
from src.mdp import TabularMDP


def evaluate(T_hat, R, gamma, true_mdp, epsilon):
    """RandomWorld has uniform initial distribution, so no single start state."""
    mdp_learned = TabularMDP(
        true_mdp.n_states, true_mdp.n_actions, T_hat, R, gamma
    )
    _, _, pi_hat = mdp_learned.compute_optimal_policy()
    _, Q_star, pi_star = true_mdp.compute_optimal_policy()

    return {
        "normalized_value": normalized_value(
            T_hat, R, gamma, true_mdp, start_state=None
        ),
        "best_matching": best_matching(pi_hat, pi_star),
        "epsilon_matching": epsilon_matching(pi_hat, Q_star, epsilon),
        "total_variation": total_variation(true_mdp.T, T_hat),
        "n_violations": count_constraint_violations(
            T_hat, R, gamma, true_mdp, epsilon
        ),
        "mse": transition_mse(true_mdp.T, T_hat),
    }


def run_one(mdp, pi_expert, coverage, epsilon, K, delta, seed, methods):
    """Generate batch data and compute T_hat for each requested method.

    Returns ``(N, T_outputs)`` where ``T_outputs`` is a dict
    ``{f"{m}_T": ndarray}`` for each ``m`` in ``methods``. Methods that
    fail (e.g. MCE solver error) get ``None``.
    """
    N, T_mle, _ = generate_batch_dataset(
        mdp, pi_expert, coverage=coverage, K=K, delta=delta, seed=seed,
    )
    out = {}
    if "mle" in methods:
        out["mle_T"] = T_mle
    # ITL is needed both as a method and as a feasibility-warm-start for BITL.
    T_itl = None
    if "itl" in methods or "bitl" in methods:
        T_itl, _ = solve_itl(
            N, T_mle, mdp.R, mdp.gamma,
            epsilon=epsilon, max_iter=10, verbose=False,
        )
    if "itl" in methods:
        out["itl_T"] = T_itl

    if "ps" in methods:
        from src.ps_baseline import ps_point_estimate
        out["ps_T"] = ps_point_estimate(N, delta=delta)

    if "mce" in methods:
        from src.mce_baseline import mce_solve
        n_states, n_actions = mdp.n_states, mdp.n_actions
        Phi = np.zeros((n_states, n_actions, n_states * n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                Phi[s, a, s * n_actions + a] = 1.0
        try:
            T_mce, _, _ = mce_solve(
                N, Phi, mdp.gamma, delta=delta,
                max_outer=3, verbose=False,
            )
            out["mce_T"] = T_mce
        except Exception:
            out["mce_T"] = None

    if "bitl" in methods:
        from src.bitl import bitl_sample
        try:
            samples, _ = bitl_sample(
                N, T_mle, mdp.R, mdp.gamma, epsilon=epsilon,
                n_samples=200, n_warmup=100,
                step_size=0.005, n_leapfrog=10, delta=delta,
                T_init=T_itl, seed=seed, verbose=False,
            )
            out["bitl_T"] = samples.mean(axis=0)
        except Exception:
            out["bitl_T"] = None

    return N, out


def main():
    print("=" * 70)
    print("  RANDOMWORLD — Benac et al. (2024) MVR (Table 4 / Figure 2 bottom)")
    print("=" * 70)

    # Paper constants
    GAMMA = 0.95
    DELTA = 0.001
    EPSILON = 5.0
    STOCHASTIC_FRACTION = float(os.environ.get("STOCHASTIC_FRACTION", "0.4"))
    K = 5                          # paper's RandomWorld K
    COVERAGES = [0.2, 0.4, 0.6, 0.8, 1.0]
    N_WORLDS = int(os.environ.get("N_WORLDS", "5"))                # paper: 20
    N_DATASETS_PER_WORLD = int(os.environ.get("N_DATASETS", "2"))  # paper: 5

    # Methods this run will compute. mle + itl always; extras opt-in.
    methods = ["mle", "itl"]
    if os.environ.get("RUN_BASELINES", "0") == "1":
        methods += ["ps", "mce"]
    if os.environ.get("RUN_BITL", "0") == "1":
        methods += ["bitl"]

    metric_keys = ["normalized_value", "best_matching", "epsilon_matching",
                    "total_variation", "n_violations", "mse"]

    os.makedirs("results/checkpoints", exist_ok=True)
    sf_tag = f"sf{int(round(STOCHASTIC_FRACTION * 100)):03d}"
    ckpt_path = f"results/checkpoints/randomworld_runs_{sf_tag}.json"
    legacy_path = "results/checkpoints/randomworld_runs.json"
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            checkpoint = json.load(f)
    elif STOCHASTIC_FRACTION == 0.4 and os.path.exists(legacy_path):
        with open(legacy_path) as f:
            checkpoint = json.load(f)
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f)
    else:
        checkpoint = {}

    rows = []
    for coverage in COVERAGES:
        per_method_runs = {m: {k: [] for k in metric_keys} for m in methods}

        for world_seed in range(N_WORLDS):
            mdp = make_randomworld(
                n_states=15, n_actions=5, gamma=GAMMA,
                n_successors=5, seed=world_seed,
            )
            pi_expert = make_epsilon_optimal_expert(
                mdp, epsilon=EPSILON,
                target_stochastic_fraction=STOCHASTIC_FRACTION,
            )

            for data_seed in range(N_DATASETS_PER_WORLD):
                key = f"{coverage}:{world_seed}:{data_seed}"
                entry = checkpoint.get(key, {})

                # Recompute only methods missing from this entry (so the
                # legacy MLE+ITL-only checkpoint is reused on this run too).
                missing = [m for m in methods if m not in entry]
                if missing:
                    combined_seed = 1000 * world_seed + data_seed
                    _N, T_outputs = run_one(
                        mdp, pi_expert, coverage, EPSILON, K, DELTA,
                        seed=combined_seed, methods=missing,
                    )
                    for m in missing:
                        T_hat = T_outputs.get(f"{m}_T")
                        if T_hat is None:
                            entry[m] = None
                            continue
                        entry[m] = evaluate(
                            T_hat, mdp.R, mdp.gamma, mdp, EPSILON
                        )
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

        n_runs = N_WORLDS * N_DATASETS_PER_WORLD
        print(f"\n  Coverage = {coverage:.1f}  (avg over {n_runs} runs)")
        for m in methods:
            nv = row.get(f"{m}_normalized_value_mean")
            bm = row.get(f"{m}_best_matching_mean")
            em = row.get(f"{m}_epsilon_matching_mean")
            if nv is None:
                continue
            print(f"    {m.upper():<5s}  NV: {nv:+.3f} ± "
                  f"{row.get(f'{m}_normalized_value_std', 0):.3f}   "
                  f"BM: {bm:.3f} ± {row.get(f'{m}_best_matching_std', 0):.3f}   "
                  f"ε-match: {em:.3f}   "
                  f"viol: {row.get(f'{m}_n_violations_mean', 0):.2f}")

    # Save
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    out_path = f"results/tables/randomworld_coverage_sweep_{sf_tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "gamma": GAMMA, "delta": DELTA, "epsilon": EPSILON,
                "stochastic_fraction": STOCHASTIC_FRACTION, "K": K,
                "n_worlds": N_WORLDS, "n_datasets_per_world": N_DATASETS_PER_WORLD,
                "coverages": COVERAGES,
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
        title=f"RandomWorld — ITL vs MLE (ε={EPSILON}, {sf_pct}% stochastic expert)",
        save_path=f"results/figures/randomworld_normalized_value_vs_coverage_{sf_tag}.png",
    )
    plot_coverage_sensitivity(
        coverages=[r["coverage"] for r in rows],
        itl_mean=[r["itl_best_matching_mean"] for r in rows],
        itl_std=[r["itl_best_matching_std"] for r in rows],
        mle_mean=[r["mle_best_matching_mean"] for r in rows],
        mle_std=[r["mle_best_matching_std"] for r in rows],
        metric_name="Best matching",
        title=f"RandomWorld — Best matching vs Coverage ({sf_pct}% stochastic)",
        save_path=f"results/figures/randomworld_best_matching_vs_coverage_{sf_tag}.png",
    )

    print(f"\n  Results written to {out_path}")
    print("  Figures in results/figures/")


if __name__ == "__main__":
    main()
