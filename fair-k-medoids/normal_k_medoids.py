"""
normal_k_medoids.py
-------------------
Normal PAM (Partitioning Around Medoids) implemented from scratch
using NumPy only.  No sklearn KMedoids is used.

PAM objective
-------------
  J(M) = sum_{x in X} min_{m in M} dist(x, m)

where dist is Euclidean distance.

A medoid must always be an actual observation from the dataset.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _euclidean_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the full (n x n) pairwise Euclidean distance matrix for X.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)

    Returns
    -------
    D : np.ndarray, shape (n, n)
        D[i, j] = ||X[i] - X[j]||_2
    """
    # Vectorised via broadcasting: avoids an explicit Python loop.
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]   # (n, n, 2)
    D = np.sqrt((diff ** 2).sum(axis=2))                 # (n, n)
    return D


# ---------------------------------------------------------------------------
# Core PAM steps
# ---------------------------------------------------------------------------

def initialize_medoids(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample k distinct medoid indices uniformly at random.

    Parameters
    ----------
    n   : int  – total number of observations.
    k   : int  – number of medoids.
    rng : np.random.Generator

    Returns
    -------
    medoid_indices : np.ndarray, shape (k,), dtype int
    """
    return rng.choice(n, size=k, replace=False)


def assign_clusters(D: np.ndarray, medoid_indices: np.ndarray) -> np.ndarray:
    """
    Assign every observation to its nearest medoid.

    Parameters
    ----------
    D             : np.ndarray, shape (n, n) – pairwise distance matrix.
    medoid_indices: np.ndarray, shape (k,)   – indices of current medoids.

    Returns
    -------
    labels : np.ndarray, shape (n,), dtype int
        labels[i] = j  means observation i is assigned to medoid_indices[j].
    """
    # Distances from every point to each medoid: shape (n, k)
    dist_to_medoids = D[:, medoid_indices]
    labels = np.argmin(dist_to_medoids, axis=1)
    return labels


def calculate_total_cost(
    D: np.ndarray,
    medoid_indices: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute J(M) = sum_i dist(x_i, m_{labels[i]}).

    Parameters
    ----------
    D             : (n, n) distance matrix.
    medoid_indices: (k,) current medoid indices.
    labels        : (n,) cluster assignment (0-indexed into medoid_indices).

    Returns
    -------
    total_cost : float
    """
    n = D.shape[0]
    assigned_medoids = medoid_indices[labels]          # shape (n,)
    costs = D[np.arange(n), assigned_medoids]          # shape (n,)
    return float(costs.sum())


def find_best_swap(
    D: np.ndarray,
    medoid_indices: np.ndarray,
) -> tuple:
    """
    Exhaustive one-medoid swap search.

    For every current medoid m_j and every non-medoid candidate c,
    evaluate J(M - {m_j} + {c}) and record the best strictly-improving swap.

    Parameters
    ----------
    D             : (n, n) distance matrix.
    medoid_indices: (k,) current medoid indices.

    Returns
    -------
    best_cost          : float  – best cost found (current cost if no improvement).
    best_medoid_indices: np.ndarray (k,) – updated medoid set (unchanged if no improvement).
    improved           : bool   – True if a strictly improving swap was found.
    """
    n = D.shape[0]
    k = len(medoid_indices)
    medoid_set = set(medoid_indices.tolist())
    non_medoids = np.array([i for i in range(n) if i not in medoid_set])

    # Baseline cost with current medoids
    labels_curr = assign_clusters(D, medoid_indices)
    best_cost = calculate_total_cost(D, medoid_indices, labels_curr)
    best_medoid_indices = medoid_indices.copy()
    improved = False

    for j in range(k):
        for c in non_medoids:
            candidate = medoid_indices.copy()
            candidate[j] = c
            labels_cand = assign_clusters(D, candidate)
            cost_cand = calculate_total_cost(D, candidate, labels_cand)
            if cost_cand < best_cost - 1e-10:
                best_cost = cost_cand
                best_medoid_indices = candidate.copy()
                improved = True

    return best_cost, best_medoid_indices, improved


def run_normal_pam(
    X: np.ndarray,
    k: int = 2,
    n_init: int = 10,
    max_iter: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run PAM with multiple random initialisations and return the best solution.

    Parameters
    ----------
    X        : (n, 2) feature matrix.
    k        : number of medoids (default 2).
    n_init   : number of random initialisations (default 10).
    max_iter : maximum accepted swaps per run (default 100).
    seed     : random seed for reproducibility (default 42).

    Returns
    -------
    result : dict with keys
        'medoid_indices' : np.ndarray (k,) – final medoid indices.
        'medoid_coords'  : np.ndarray (k, 2) – actual coordinates.
        'labels'         : np.ndarray (n,) – cluster labels (0 … k-1).
        'total_cost'     : float – final total distance.
        'n_swaps'        : int – number of accepted swaps in winning run.
        'runtime_s'      : float – wall-clock seconds for winning run.
    """
    n = X.shape[0]
    D = _euclidean_distance_matrix(X)
    rng = np.random.default_rng(seed)

    best_overall_cost = np.inf
    best_overall_result = None

    for init_idx in range(n_init):
        t0 = time.time()
        medoid_indices = initialize_medoids(n, k, rng)
        n_swaps = 0

        for _ in range(max_iter):
            _, new_medoids, improved = find_best_swap(D, medoid_indices)
            if not improved:
                break
            medoid_indices = new_medoids
            n_swaps += 1

        labels = assign_clusters(D, medoid_indices)
        total_cost = calculate_total_cost(D, medoid_indices, labels)
        runtime = time.time() - t0

        if total_cost < best_overall_cost:
            best_overall_cost = total_cost
            best_overall_result = {
                'medoid_indices': medoid_indices.copy(),
                'medoid_coords':  X[medoid_indices].copy(),
                'labels':         labels.copy(),
                'total_cost':     total_cost,
                'n_swaps':        n_swaps,
                'runtime_s':      runtime,
            }

    return best_overall_result
