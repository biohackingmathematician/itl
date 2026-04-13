# ITL Reproduction: Inverse Transition Learning (Benac et al., 2024)

Reproducing the results from "Inverse Transition Learning: Learning Dynamics from Demonstrations" as the foundation for Constrained ITL (C-ITL) thesis work.

## What ITL Does

Standard RL: knows transition dynamics T, learns optimal policy pi*.
ITL: knows reward R and observes expert demonstrations, learns transition dynamics T*.

The key insight is that an expert's near-optimal behavior constrains what the environment dynamics can be. If the expert always goes Right instead of Left, the dynamics must make Right sufficiently better — otherwise the expert's behavior is irrational.

## Experiments to Reproduce

### 1. Gridworld (Synthetic)
- **Input**: 5x5 grid, 25 states, 4 actions (up/down/left/right), known reward R, discount gamma=0.9
- **Data**: Simulated expert trajectories (N transitions per state-action pair), fully synthetic
- **Output**: Estimated T_hat via ITL QP, MSE vs true T*, comparison to MLE baseline
- **Target**: Reproduce Table 1 (MSE comparison) and Figure 1 (epsilon sensitivity)

### 2. RandomWorld (Synthetic)
- **Input**: Randomly generated MDP, 15 states, 5 actions, random transition matrices
- **Data**: Simulated expert trajectories, fully synthetic
- **Output**: Same as Gridworld — MSE comparison, epsilon sensitivity
- **Target**: Reproduce Table 1 RandomWorld columns

### 3. MIMIC-IV (Real Clinical Data)
- **Input**: Sepsis patient trajectories from MIMIC-IV database
- **States**: Discretized patient vitals (heart rate, BP, lactate, etc.)
- **Actions**: Treatment decisions (vasopressors, IV fluids — discretized)
- **Reward**: Clinical outcome (mortality-based)
- **Data**: Real clinical data, requires PhysioNet credentialed access
- **Output**: Learned hospital transition dynamics, outlier trajectory detection, Bayesian Regret for reward transfer
- **Target**: Reproduce Section 5.3 results

### 4. BITL (Bayesian Extension)
- **Input**: Same environments as above
- **Output**: Posterior distribution over T* (not just point estimate), outlier detection scores, Bayesian Regret
- **Method**: HMC with reflection off constraint boundaries (Algorithm 3 in paper)

## Project Structure

```
itl-reproduction/
├── src/
│   ├── mdp.py              # MDP definition, value iteration, Q-value computation
│   ├── itl_solver.py       # Core ITL QP solver (CVXPY)
│   ├── bitl.py             # Bayesian ITL (HMC sampling from constrained posterior)
│   ├── expert.py           # Expert policy simulation and data generation
│   ├── environments.py     # Gridworld, RandomWorld environment definitions
│   └── utils.py            # Metrics (MSE), plotting, data loading
├── experiments/
│   ├── run_gridworld.py    # Full Gridworld reproduction
│   ├── run_randomworld.py  # Full RandomWorld reproduction
│   └── run_mimic.py        # MIMIC-IV pipeline
├── notebooks/
│   ├── 01_verify_corridor.ipynb   # Verify solver on 3-state example from notes
│   ├── 02_gridworld_results.ipynb # Gridworld analysis and figures
│   └── 03_mimic_eda.ipynb         # MIMIC-IV data exploration
├── data/                   # gitignored — downloaded/generated data goes here
├── results/
│   ├── figures/
│   └── tables/
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Verify on 3-state corridor (should match hand calculations)
python -m experiments.run_corridor

# Full Gridworld reproduction
python -m experiments.run_gridworld

# Full RandomWorld reproduction
python -m experiments.run_randomworld
```

## Key Dependencies
- cvxpy: Convex optimization (QP solver for ITL)
- numpy, scipy: Linear algebra, matrix operations
- matplotlib, seaborn: Plotting
- pandas: Data handling (esp. MIMIC-IV)
