"""
PS = Posterior Sampling baseline (paper Table 4 column).

This is the unconstrained Bayesian baseline that BITL is supposed to beat.
Same Dirichlet-Categorical posterior over T(.|s,a) rows as BITL, but WITHOUT
the eps-ball constraints from Eq 8 / Eq 9. So the only information used is
the empirical counts N — none of the "the expert is eps-optimal" prior gets
in.

PS gives:
  - posterior mean → point estimate (essentially T_MLE shifted by Dir prior)
  - posterior samples → CVaR / Bayesian regret

Cross-references:
  Krause & Hubotter Ch 11 Bayesian model-based RL: this is the textbook
  Dir-Categorical posterior with no constraint, the same starting point that
  BITL refines with Eq 8/9.
  Osband et al. 2013 "Posterior Sampling for RL (PSRL)": the original
  motivation for this baseline in the broader RL literature.

Note on the paper: "PS" appears as a Table 4 column without a clear method
description in the parts of the paper text I have. This implementation is
the most natural reading — unconstrained Dirichlet posterior — which also
matches what would beat MLE in tables 4-6 by virtue of Bayesian smoothing.
Verify against paper Section 5 / appendix when PDF is re-uploaded.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def ps_sample(
    N: np.ndarray,
    delta: float = 0.001,
    n_samples: int = 500,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Sample K transition matrices from Dir(N + delta) per (s, a). No
    constraints, no HMC — just direct Dirichlet sampling.

    Args:
        N: transition counts, shape (n_states, n_actions, n_states).
        delta: Dirichlet pseudo-count (paper Eq 5 default 0.001).
        n_samples: K.
        seed: RNG seed.

    Returns:
        samples: shape (K, n_states, n_actions, n_states).
        info: dict with metadata (delta, n_samples, etc.).
    """
    rng = np.random.default_rng(seed)
    n_states, n_actions, _ = N.shape
    samples = np.zeros((n_samples, n_states, n_actions, n_states))
    alpha = N + delta
    for s in range(n_states):
        for a in range(n_actions):
            samples[:, s, a, :] = rng.dirichlet(alpha[s, a], size=n_samples)

    info = {
        "delta": delta,
        "n_samples": n_samples,
        "method": "PS (unconstrained Dir-Categorical)",
    }
    return samples, info


def ps_point_estimate(N: np.ndarray, delta: float = 0.001) -> np.ndarray:
    """
    Posterior mean point estimate: mean of Dir(N + delta) per (s, a).
    This is exactly the Laplace-smoothed MLE (paper Eq 5), same as T_MLE,
    but reported as a separate baseline column to contrast with BITL's
    constrained posterior.
    """
    smoothed = N + delta
    return smoothed / smoothed.sum(axis=2, keepdims=True)
