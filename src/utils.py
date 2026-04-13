"""
Metrics, plotting, and utility functions for ITL experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def transition_mse(T_true: np.ndarray, T_hat: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Mean squared error between true and estimated transitions.

    Args:
        T_true: true transitions, shape (n_states, n_actions, n_states)
        T_hat: estimated transitions, shape (n_states, n_actions, n_states)
        mask: optional boolean mask of shape (n_states, n_actions).
              If provided, only compute MSE over masked (s, a) pairs.

    Returns:
        mse: mean squared error
    """
    if mask is not None:
        errors = []
        for s in range(T_true.shape[0]):
            for a in range(T_true.shape[1]):
                if mask[s, a]:
                    errors.append(np.sum((T_true[s, a] - T_hat[s, a]) ** 2))
        if len(errors) == 0:
            return 0.0
        return np.mean(errors)
    else:
        return np.mean(np.sum((T_true - T_hat) ** 2, axis=2))


def transition_mse_visited_vs_unvisited(
    T_true: np.ndarray,
    T_hat: np.ndarray,
    N: np.ndarray,
) -> dict:
    """
    Compute MSE separately for visited and unvisited (s, a) pairs.

    This is the key metric in Table 1 of the paper: ITL should beat MLE
    on unvisited pairs (where MLE has no data but ITL uses constraints).

    Returns:
        dict with keys: 'mse_all', 'mse_visited', 'mse_unvisited', 'n_visited', 'n_unvisited'
    """
    sa_visited = N.sum(axis=2) > 0  # (n_states, n_actions)

    return {
        "mse_all": transition_mse(T_true, T_hat),
        "mse_visited": transition_mse(T_true, T_hat, mask=sa_visited),
        "mse_unvisited": transition_mse(T_true, T_hat, mask=~sa_visited),
        "n_visited": sa_visited.sum(),
        "n_unvisited": (~sa_visited).sum(),
    }


def plot_epsilon_sensitivity(
    epsilons: list,
    mse_itl: list,
    mse_mle: list,
    title: str = "ITL vs MLE: Epsilon Sensitivity",
    save_path: Optional[str] = None,
):
    """
    Plot MSE vs epsilon for ITL and MLE baseline.
    Reproduces Figure 1 from the paper.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(epsilons, mse_itl, "o-", label="ITL", color="#2563eb", linewidth=2)
    ax.axhline(y=mse_mle[0] if isinstance(mse_mle, list) else mse_mle,
               color="#dc2626", linestyle="--", label="MLE", linewidth=2)

    ax.set_xlabel("Epsilon (near-optimality tolerance)", fontsize=12)
    ax.set_ylabel("MSE (transition estimate)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_transition_heatmap(
    T: np.ndarray,
    action: int,
    title: str = "Transition Matrix",
    save_path: Optional[str] = None,
):
    """
    Plot transition matrix T[:, action, :] as a heatmap.
    Useful for comparing true vs estimated dynamics visually.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    im = ax.imshow(T[:, action, :], cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Next State s'", fontsize=12)
    ax.set_ylabel("Current State s", fontsize=12)
    ax.set_title(f"{title} (action={action})", fontsize=14)
    plt.colorbar(im, ax=ax, label="P(s'|s,a)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_results_table(results: dict, method_name: str = "ITL"):
    """Print a clean results table matching the paper's Table 1 format."""
    print(f"\n{'='*60}")
    print(f"  {method_name} Results")
    print(f"{'='*60}")
    print(f"  MSE (all pairs):        {results['mse_all']:.6f}")
    print(f"  MSE (visited pairs):    {results['mse_visited']:.6f}")
    print(f"  MSE (unvisited pairs):  {results['mse_unvisited']:.6f}")
    print(f"  Visited (s,a) pairs:    {results['n_visited']}")
    print(f"  Unvisited (s,a) pairs:  {results['n_unvisited']}")
    print(f"{'='*60}\n")
