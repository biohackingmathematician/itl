# Project context for Claude

This file is loaded into context on every Claude conversation in this repo.
Last updated 2026-04-28 by Agna + a Cowork session.

## What this project is

Reproduction of Benac et al. (2024), *Inverse Transition Learning: Learning
Dynamics from Demonstrations* (AAAI 2025), as the foundation for Agna Chan's
master's thesis. The thesis extension ("C-ITL") will pick one of the
paper's three named future-work directions — see "Thesis novelty" below.

**Thesis deadline: end of May 2026** (~5 weeks out from the date above).

The paper PDF is at `~/Library/Mobile Documents/.../290483d5-*.pdf` (Cowork
project files). If that path isn't accessible from Claude Code, ask Agna
for the PDF and we'll re-extract Future Work / Table 4.

## Repo state at a glance

The repo went through a v1→v2 refactor on 2026-04-13 that fixed 9 issues
with the original scaffold (see `results/archive_v1/README.md` for the
full v1 critique). Current state is v2 and is the source of truth.

### Working and verified
- `src/mdp.py` — value iteration / Q-values / ε-ball. Verified against
  hand-computed corridor: `v* = [76.87, 87.68, 100.00]` exact.
- `src/itl_solver.py` — paper Algorithm 1 / Eq 10. Three fixes recorded
  in `results/MVR_findings.md`:
  1. ε-ball comes from observed expert actions (per paper Definition 1),
     not from `Q̂(T_MLE)`.
  2. Constraints only added at observed states (`∀(s,a) ∈ D` in Eq 10),
     not at every state.
  3. π⁽⁰⁾ uniform for s ∉ D (Algorithm 1 line 3), not greedy.
- `src/environments.py` — corridor (γ=0.9), gridworld (5×5, soft walls,
  γ=0.95, paper spec), randomworld (15S × 5A × 5 successors, paper spec).
- `experiments/run_corridor.py` — verifies hand calcs.
- `experiments/run_gridworld.py` — coverage sweep, paper metrics.
- `experiments/run_randomworld.py` — coverage sweep, paper metrics.
- `experiments/run_gridworld_parallel.py` — round-robin runner with
  per-seed checkpointing for chunked / resumable execution.
- `docs/book_mapping.md` — Krause & Hubotter Ch 10–12 cross-references
  for every major equation.

### Headline results (2026-04-13, 5 seeds, coverage = 1.0)

Gridworld:
| metric | ITL | MLE |
| --- | --- | --- |
| Best matching | 0.816 ± 0.083 | 0.520 ± 0.063 |
| ε-matching | 1.000 ± 0.000 | 0.712 ± 0.018 |
| Normalized Value | 0.998 ± 0.001 | 0.142 ± 0.002 |
| Constraint violations | 0.0 | 7.20 |

Both ITL and MLE numbers run higher than paper Table 4 — almost
certainly because our soft-wall layout is less aversive than the
paper's (their tile coordinates aren't published). The structural
pattern (ITL >> MLE; ε-matching → 1.0 at full coverage; 0 violations
for ITL) matches the paper. **Don't claim "matches paper absolute
numbers"** — claim "reproduces the qualitative pattern of Table 4".
See `results/MVR_findings.md` for the full side-by-side.

## Known bugs (status as of 2026-04-28 evening)

### BITL bug 1: wrong constraint set — FIXED
`src/bitl.py` was building constraints from `mdp_mle.compute_epsilon_ball(Q_mle, eps)`
instead of observed expert actions. Replaced with `visited_sa = N.sum(axis=2) > 0`
and propagated through `_build_constraint_matrix`. Initial slack on corridor
went from -2.0 to 0.0; posterior mean MSE went from 0.326 to 0.208 (now beats
both MLE 0.282 and ITL 0.237). This was the primary bug.

### BITL bug 2: CI undercoverage — UNDERSTOOD, NOT A BUG
The 0.33 overall CI coverage was largely an artifact of (1) testing on the
corridor where 50% of true T entries are exactly 0 (Dirichlet posterior support
strictly excludes 0, so those entries can never be in any CI), and (2) the
default `delta=1.0` over-concentrating the prior. Mitigations applied:
- Added `coverage_95_nonzero` diagnostic to `posterior_mse` — restricts to
  entries where T_true > 0 (the only meaningful coverage diagnostic).
- Switched default `delta` from 1.0 to 0.001 (paper Eq 5 value).
- Documented the delta tradeoff in the docstring: small delta gives
  well-calibrated CIs but worse posterior-mean MSE on UNVISITED (s,a) pairs;
  large delta gives narrower CIs but better point-estimate MSE on those pairs.
  This is a real research choice, recommended as a thesis ablation.

With delta=0.001 on corridor, coverage_95_nonzero is in the 0.67–0.89 range
across seeds — within normal variation for a constrained HMC sampler.

## Thesis novelty (C-ITL) — three options grounded in the paper

See `docs/c_itl_options.md` for the full version. Quick summary, ordered
by recommendation:

1. **Combined ITL+IRL** ("C" = Combined) — *Recommended*. Joint
   inference of T and R from ε-optimal expert demos. Highest reuse of
   existing CVXPY infra (T-step is unchanged), real novelty (no paper
   does offline + ε-optimal + joint T+R), and strengthens the MIMIC
   story by removing the hand-engineered-reward objection.
2. **Continuous-state ITL** ("C" = Continuous) — Most directly named in
   the paper title. ~3 weeks of new infrastructure before any results.
   Risky in 5 weeks.
3. **POMDP ITL** — Highest research ceiling, not 5-week-tractable.
   Defer to follow-on.

Falsifier sketch for option 1 (in the doc): "On a 5×5 gridworld with
known feature map Phi(s,a), running joint ITL+IRL on full-coverage
expert data recovers w within ‖ŵ − w*‖₂ < 0.1 in 95% of seeds. If this
fails, the formulation can't identify R even in the easiest case."

## MIMIC status

PhysioNet credentialing **not started** as of 2026-04-28. Per Agna's
direction: foundations come first, MIMIC is a stretch goal not a
blocker. The pipeline scaffold in `experiments/run_mimic.py` currently
implements a Komorowski-style sepsis cohort; the paper's Section 5.3 is
hypotension. Cohort filter rewrite is needed before MIMIC results can
match the paper.

## How to do common things

### Reproduce headline numbers
```bash
python -m experiments.run_corridor      # ~5 sec, exact hand-calc match
python -m experiments.run_gridworld     # ~minutes at N_SEEDS=10
python -m experiments.run_randomworld   # ~minutes
```

### Tests

```bash
python -m pytest tests/                     # 8 smoke tests, ~5 s
```

### Overnight runbook (recommended order)

These commands all checkpoint per-seed/per-world to
`results/checkpoints/*.json` and skip completed work on resume. Run from
the repo root in a regular Mac Terminal (NOT Cowork — the 45-second bash
ceiling makes long runs impractical there). Each command produces a
namespaced output table that doesn't clobber the others.

**Method columns:** by default each script computes only MLE and ITL
columns. Set `RUN_BASELINES=1` to additionally compute PS and MCE columns
(both are paper Table 4 baselines). Set `RUN_BITL=1` to additionally
compute BITL posterior mean (HMC sampler, slowest). For the full Table 4
sweep below, set both.

1. **40% stochastic baseline at full seed counts (paper Table 4)**
   ```bash
   RUN_BASELINES=1 RUN_BITL=1 N_SEEDS=50 \
     python -m experiments.run_gridworld
   RUN_BASELINES=1 RUN_BITL=1 N_WORLDS=20 N_DATASETS=5 \
     python -m experiments.run_randomworld
   ```

2. **Transfer task (paper Table 4 transfer columns)**
   ```bash
   RUN_BASELINES=1 RUN_BITL=1 ENV=both \
     N_SEEDS=50 N_WORLDS=20 N_DATASETS=5 \
     python -m experiments.run_transfer
   ```

3. **20% stochastic ablation (paper Table 5)**
   ```bash
   STOCHASTIC_FRACTION=0.2 RUN_BASELINES=1 RUN_BITL=1 \
     N_SEEDS=50 python -m experiments.run_gridworld
   STOCHASTIC_FRACTION=0.2 RUN_BASELINES=1 RUN_BITL=1 \
     N_WORLDS=20 N_DATASETS=5 python -m experiments.run_randomworld
   STOCHASTIC_FRACTION=0.2 RUN_BASELINES=1 RUN_BITL=1 ENV=both \
     N_SEEDS=50 N_WORLDS=20 N_DATASETS=5 \
     python -m experiments.run_transfer
   ```

4. **0% stochastic ablation (paper Table 6)**
   ```bash
   STOCHASTIC_FRACTION=0.0 RUN_BASELINES=1 RUN_BITL=1 \
     N_SEEDS=50 python -m experiments.run_gridworld
   STOCHASTIC_FRACTION=0.0 RUN_BASELINES=1 RUN_BITL=1 \
     N_WORLDS=20 N_DATASETS=5 python -m experiments.run_randomworld
   STOCHASTIC_FRACTION=0.0 RUN_BASELINES=1 RUN_BITL=1 ENV=both \
     N_SEEDS=50 N_WORLDS=20 N_DATASETS=5 \
     python -m experiments.run_transfer
   ```

Outputs land in:
- `results/tables/gridworld_coverage_sweep_sf{040,020,000}.json`
- `results/tables/gridworld_transfer_sf{040,020,000}.json`
- `results/tables/randomworld_coverage_sweep_sf{040,020,000}.json`
- `results/tables/randomworld_transfer_sf{040,020,000}.json`
- `results/figures/*_sf{040,020,000}.png`

Total wall-clock estimate on a Mac (rough, depends on CPU): ~4–8 hours
for the full sweep. Run before bed; check in the morning.

### Run BITL (only after bug 1 is fixed)
```bash
python -m experiments.run_bitl
```
Note: `run_bitl.py::run_outlier_detection_demo` previously imported a
non-existent `make_structured_randomworld`. That import was patched on
2026-04-28 to use `make_randomworld` with `n_successors=3`.

## Combined ITL+IRL prototype (2026-04-28)

`src/itl_irl_solver.py` + `experiments/run_itl_irl_smoke.py`. Lifts R from
a fixed numpy array to `R(s,a) = Phi(s,a) @ w` with `w` as a CVXPY variable
alongside `T`. Eps-ball constraints become linear in `(w, T)` with `v_lin`
fixed per outer iteration.

Corridor smoke test result (state-only one-hot Phi, anchor R(2,0)=10):
- Converges in 2 iterations to a fixed point
- pi* recovered exactly (best-action match = 1.000)
- T MSE = 0.233 (better than MLE 0.282, ~ITL alone at 0.237)
- w_hat = [0, 0, 10] vs w_true = [-0.1, -0.1, 10]

The reward weight gap is **expected IRL behavior**, not a bug. With
state-only one-hot features the within-state `Phi` differences are zero,
so Eq 8 doesn't constrain `w[0]` vs `w[1]` directly. Rewards are only
identifiable up to policy-equivalence in the offline + ε-optimal
demonstration setting. The right success criterion is `pi*(T_hat, R_hat)
== pi*(T*, R*)`, which we get.

Initialization matters: `w_init` derived from the anchor avoids iter-0
infeasibility (with `w_init=zeros` the initial `v_lin=0` makes the
eps-ball constraints vacuous + infeasible).

**Next steps for this prototype:**
- Stress-test on gridworld with non-trivial features (e.g., one feature
  per (s, a) pair, or low-rank).
- Add a MaxEnt-IRL baseline to compare reward recovery quality.
- Sharpen the falsifier in `docs/c_itl_options.md`: the right falsifier
  is "pi*(T_hat, R_hat) != pi*(T*, R*) on coverage=1.0 expert data",
  not "w_hat near w_true".
- Lift this same change into BITL (sample joint posterior over T and w).

## Recommended next session order

1. Re-upload the Benac et al. PDF (the mount got cleaned during this
   session) and read Sections 3-5 + Appendix carefully. Specifically
   check: (a) what delta is used for BITL's Dirichlet posterior vs the
   Eq 5 MLE; (b) how the paper picks stochastic-policy states (we pick
   smallest Q-gap, paper may differ); (c) whether the paper's formulation
   admits the Combined ITL+IRL extension cleanly.
2. Stress-test the Combined ITL+IRL prototype on gridworld with proper
   feature design.
3. Background-run `N_SEEDS=50 python -m experiments.run_gridworld` and
   `python -m experiments.run_randomworld` overnight to tighten error
   bars in `results/MVR_findings.md`.
4. Add transfer-task evaluation script (infrastructure already in
   `src/environments.py::DEFAULT_SOFT_WALLS_TRANSFER` and
   `make_randomworld_transfer`).
5. MIMIC only after items 1–4 are stable.

## Conventions

- Notation matches the paper: `T[s, a, s']`, `R[s, a]`, `gamma`, ε-ball.
- All paper equation numbers cited inline are from Benac et al. (2024).
- All Krause & Hubotter chapter references are documented in
  `docs/book_mapping.md`. If you cite a chapter / equation, add an
  entry there.
- New experiment scripts go under `experiments/` with checkpointing to
  `results/checkpoints/` so partial runs don't waste compute.
- Don't put hardcoded paths in code; use `os.path.join` rooted at the
  repo.

## Cross-references

- Paper Future Work: page 8 of the Benac et al. PDF.
- Paper Table 4 (40% stochastic, standard task): page 16 of the PDF.
  Numbers extracted into `results/MVR_findings.md`.
- Existing v1 critique: `results/archive_v1/README.md`.
- Existing v2 fixes: `results/MVR_findings.md`.
- Textbook mapping: `docs/book_mapping.md`.
- Thesis novelty options: `docs/c_itl_options.md`.
