# C-ITL: thesis novelty options grounded in the paper

Drafted 2026-04-28. Per Agna's direction: stick strictly to what Benac et
al. (2024) themselves flag as future work. No invented extensions.

## What the paper says, verbatim

From Benac et al. (2024), end of Discussion / Future Work (page 8 of the
PDF in `.projects/.../files/290483d5-*.pdf`):

> While our method provides a robust approach for estimating environmental
> dynamics using expert demonstrations, it is currently limited to discrete
> and fully observable state spaces. Future research could explore extending
> ITL to handle more complex environments, including those with
> high-dimensional, continuous, or partially observable state spaces.
> Additionally, combining ITL with Inverse Reinforcement Learning (IRL) to
> simultaneously learn rewards and dynamics represents another promising
> avenue for future work.

That paragraph names exactly three directions:

1. Continuous (or high-dimensional) state spaces
2. Partially observable state spaces (POMDPs)
3. Combined ITL + IRL — joint learning of T* and R from demonstrations

These are the only directions sourced from the paper itself. Everything
below assesses tractability strictly against the 5-week deadline (May 31,
2026), assuming we want polished synthetic foundations *plus* a thesis
contribution.

## Direction 1 — Continuous-state ITL ("C" = Continuous)

The cleanest match to the paper title and the most directly named in the
Future Work paragraph.

### What it would look like
- Replace the per-(s,a) Dirichlet over discrete next-state distributions
  with a parametric model T_theta(s' | s, a). Simplest tractable choice:
  linear-Gaussian dynamics s' = A_a s + b_a + N(0, Sigma_a).
- Recast Eq 8/9 of the paper using V^pi as a function of theta. The
  v_lin vector becomes a function evaluation at sampled states, not a
  fixed (n_states,) vector.
- ε-ball constraints are enforced at the *finite* (s, a) pairs in the
  expert dataset, not at every state.
- Existing CVXPY pipeline does not survive. The QP becomes a non-convex
  optimization in theta. Alternation no longer trivially converges.

### What's hard in 5 weeks
- All-new optimizer (likely SGD on a smooth objective with Lagrangian
  penalties, or trust-region with constraint linearization).
- Pick a continuous-state benchmark (mountain car, cartpole, pendulum)
  and re-define the eps-ball at sampled states, which is ill-defined
  unless we discretize on the fly.
- No baseline carries over from the discrete code.

### 5-week verdict
HIGH risk. Best case: ~3 weeks of new infrastructure, ~1 week of
experiments, ~1 week of writeup — cuts very close to the deadline with
no slack for things going wrong. Cleanly scoped to "linear-Gaussian only"
this could land, but the contribution is then "ITL works in a setting
where the dynamics are already easy to learn parametrically", which is a
thinner thesis than the full continuous claim.

## Direction 2 — POMDP ITL

The paper names "partially observable state spaces" but does not develop
the idea further.

### What it would look like
- States are not directly observed; instead we observe O ~ p(. | S).
- The ε-ball is defined over policy-relevant *belief* states, not raw
  states.
- Need to estimate (T, p(O|S)) jointly from demonstrations of (O, A, O').

### 5-week verdict
HIGHEST risk. POMDP RL is its own subfield with well-known identifiability
and computational issues. ITL+POMDP is genuinely an open problem with
high research ceiling but not 5-week-tractable for a master's thesis.
Defer.

## Direction 3 — Combined ITL+IRL ("C" = Combined)

The paper assumes R is known. This is a strong assumption — particularly
in healthcare, where R is hand-engineered (in MIMIC: ±15 for survival /
mortality is a survival proxy, not a real reward signal). Joint learning
of T and R from eps-optimal expert demonstrations is the IRL + ITL
combination the paper flags.

### What it would look like
- Decision variables are (T, R) instead of just T.
- The eps-ball constraint from Eq 8/9 becomes
  ```
  R(s, a) - R(s, a') + gamma * (T_sa - T_sa').T @ v_lin >= epsilon
  ```
  where now BOTH R and T are unknown. This is bilinear in (R, T) once v_lin
  also depends on R (which it does, via v_lin = (I - gamma P^pi)^-1 R^pi).
- An identifiability prior on R is required. The standard MaxEnt-IRL
  literature (Ziebart et al. 2008) and follow-ups handle this with
  features and a sparsity / regularization assumption. We can adopt the
  cleanest one: R = Phi(s, a)^T @ w with sparsity prior on w.

### Why this is the sharpest 5-week fit
- The infrastructure overlap with our existing code is high. The CVXPY
  QP can be extended to include R variables and a reward-feature matrix,
  with the bilinear term linearized via the existing alternation scheme.
  Concretely: hold R fixed, solve for T (existing QP); hold T fixed,
  solve for R (a smaller QP with eps-optimality constraints).
- The MIMIC story benefits enormously: we no longer need to hand-engineer
  a survival proxy reward; we infer R from clinician decisions. This
  makes the MIMIC chapter (if we get there) defendable against
  "your reward function is arbitrary".
- The novelty is real. No paper in our citations does joint T+R from
  eps-optimal experts in the offline setting. Closest priors:
  - Ziebart et al. 2008 MaxEnt-IRL: known T, learns R
  - Chen et al. 2020 offline IRL: known T or learned with separate model
  - Benac et al. 2024 ITL: known R, learns T (this paper)
  - Combined: neither known, both inferred jointly under eps-optimality

### What's hard in 5 weeks
- Identifiability writeup. (T, R, gamma) is jointly underdetermined
  without prior assumptions; the thesis has to argue precisely what
  identifies R. The cleanest version is "R has known feature
  representation Phi, learn weights w under L1 sparsity prior".
- Comparison baselines: at least MaxEnt-IRL with a model-free T estimate
  needs to be reproduced for honest comparison.
- The QP becomes alternating (T-step, R-step) instead of the existing
  paper's alternation (which is policy-step, T-step internally). Need
  to verify convergence.

### 5-week verdict
MEDIUM risk. Higher novelty ceiling than Direction 1, more defensible
thesis story (especially with MIMIC), and more reuse of existing code.
The identifiability section is the main writeup load.

## Recommendation grounded in the paper

If the directive is "polished synthetic foundation + try MIMIC + thesis
novelty grounded in the paper", **Direction 3 (Combined ITL+IRL) is the
sharpest fit**. It:

1. Comes verbatim from the paper's named future work.
2. Reuses the existing CVXPY infrastructure (T-step is unchanged).
3. Strengthens the MIMIC story by removing the "but R is hand-engineered"
   objection — IF we get to MIMIC at all.
4. Has clear novelty (no paper combines offline + eps-optimal + joint T+R).
5. Has a clean falsifier: "if the joint posterior on (T, R) does not
   identify R within an L2 ball of the true R as N → ∞ for the synthetic
   benchmark, the formulation does not work."

Direction 1 (Continuous) is the second-best paper-grounded option but
costs ~3 weeks of pure infrastructure before any results, which is too
risky given the BITL bugs we still need to fix and the seed sweep that's
still pending.

Direction 2 (POMDP) is too ambitious for a master's thesis at this
deadline.

## What we'd actually need to write

For Direction 3, the math sketch:

1. **Decision variables**: w ∈ R^d (reward weights), T_{s,a} ∈ Δ^|S| for
   each (s, a). Reward features Phi(s, a) ∈ R^d are fixed inputs.
2. **Reward**: R(s, a) = Phi(s, a)^T @ w.
3. **Objective**: weighted L2 fit to T_MLE on T (paper Eq 10) PLUS L1
   sparsity penalty lambda * ||w||_1 on w.
4. **Constraints**: same eps-ball Eq 8/9 from the paper, but now both
   sides depend on (w, T).
5. **Solver**: alternate between (T-step: hold w fixed, solve paper's
   QP) and (R-step: hold T fixed, solve a smaller LP/QP for w under
   eps-optimality constraints with v_lin treated as a function of w).
6. **Identifiability**: prove that under L1 sparsity and assumption
   that the expert visits a sufficiently rich subset of states, the
   joint optimum (T*, w*) is unique up to scaling.

That's a thesis chapter's worth of work, but tractable in 4 weeks if
infrastructure starts now and BITL bugs are fixed in parallel.

## Methodology gap discovered 2026-05-01: goal-domination

The Combined ITL+IRL prototype hits best-action match = 1.000 on the 5×5
gridworld (commit `2ff8540`), but a follow-up sanity check showed this is
mostly an artifact of the benchmark, not a method success.

### The check

On the same gridworld, paired with `T_MLE` (no ITL, no IRL):
- `pi*(T_MLE, R_TRUE)` matches `pi*(T*, R*)` at 0.480.
- `pi*(T_MLE, R_trivial)` where `R_trivial = 10 * 1[s == goal]` (no step
  cost, no soft-wall penalty) also matches at 0.480.

So a degenerate goal-only reward function paired with a noisy `T_MLE`
gives the same policy quality as the *true* reward paired with the same
`T_MLE`. The reward structure (the −0.1 step costs, the −5 soft-wall
penalties) doesn't matter for the optimal policy as long as the goal is
correctly identified.

### What this means for the thesis

The current "match = 1.000" success criterion for ITL+IRL on gridworld
is a *necessary but not sufficient* test for joint reward+dynamics
recovery. To certify that the IRL step is doing useful work — and not
just riding ITL's improvement on T plus a goal-located reward — the
benchmark must require the reward *structure*, not just the goal
location, to recover the optimal policy.

### Proposed non-goal-dominated benchmark

A 5×5 gridworld with two competing terminal cells:
- Goal A at (0, 4): R = +5
- Goal B at (4, 0): R = +10
- Step cost: −0.1
- One soft-wall barrier preventing the obvious diagonal path

The optimal policy depends on the magnitude of `R_A` vs `R_B`. If a
reward learner recovers a goal-only signal "+10 at A, +10 at B", the
policy is ambiguous — half the seeds will converge to A, half to B.
Only a learner that recovers the magnitude *difference* between the
two goals will reliably match `pi*`.

For ITL+IRL to claim it learns R, the falsifier becomes:
> On the two-goal gridworld with `R(A) = 5`, `R(B) = 10`, ITL+IRL with
> one-hot state-action features and a single anchor at goal B must
> recover `pi*` significantly more often than:
> (a) MLE-T + R_trivial = `10 * 1[s == goal_A or s == goal_B]`
> (b) ITL-T + R_trivial.
> If neither (a) nor (b) recovers `pi*` reliably and ITL+IRL does, the
> IRL step is doing real work. If all three recover `pi*` equally, the
> benchmark is still goal-dominated and we need a harder one.

### Implementation plan

1. Add a `make_two_goal_gridworld(...)` helper to `src/environments.py`.
2. Add `experiments/run_itl_irl_two_goal.py` mirroring the existing
   stress test.
3. Write a unit test that asserts the trivial-R policy is *worse* than
   the true-R policy on this benchmark — so we know the benchmark
   actually distinguishes them before we run the full IRL comparison.
4. Add to the thesis chapter: this is the experiment that makes the
   joint-inference claim defensible.

This is a 1-2 day add. Higher value than running more seeds on the
existing goal-dominated gridworld.

## Quick falsifiers (next session)

Before writing more code, sketch a one-paragraph falsifier per direction.
For Direction 3:

> "On a 5×5 gridworld with known feature map Phi(s, a) ∈ R^25 (one-hot
> over states), running joint ITL+IRL on full-coverage expert data
> recovers w within ||w_hat - w_true||_2 < 0.1 in 95% of seeds. If this
> fails, the formulation cannot identify R even in the easiest case."

If that falsifier is hard to specify or hard to test, the contribution
isn't sharp enough yet — go back and tighten.

## Cross-references
- Paper Future Work: page 8 of `290483d5-*.pdf`.
- Existing ITL QP: `src/itl_solver.py::_solve_qp` lines 264–393. The
  R-variable extension would slot in by lifting R into a `cp.Variable`
  with a `Phi @ w` parameterization.
- Existing BITL: `src/bitl.py::_build_constraint_matrix`. Same lift
  applies; adds a Dirichlet posterior over T plus a Gaussian / Laplace
  posterior over w.
