"""
fair_k_medoids.py
-----------------
Fair-PAM: a fairness-aware Partitioning Around Medoids heuristic.

Fairness objective
------------------
  GA(M) = (1 / |A|) * sum_{x in A} min_{m in M} dist(x, m)
  GB(M) = (1 / |B|) * sum_{x in B} min_{m in M} dist(x, m)
  F(M)  = max(GA(M), GB(M))

Swap selection is based on minimising F(M), NOT the total distance.
Cluster assignment is still nearest-medoid (same as normal PAM).

Group labels are used only to compute GA, GB and F.
They are never used as clustering features.
"""

import time
import numpy as np

# Reuse distance/assignment helpers from normal PAM
from normal_k_medoids import (
    _euclidean_distance_matrix,
    initialize_medoids,
    assign_clusters,
)


# ---------------------------------------------------------------------------
# Fair-objective helpers
# ---------------------------------------------------------------------------

def calculate_group_costs(
    D: np.ndarray,
    medoid_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple:
    """
    Compute per-group average nearest-medoid distances.

    Parameters
    ----------
    D             : (n, n) pairwise distance matrix.
    medoid_indices: (k,) current medoid indices.
    labels        : (n,) cluster assignment (0-indexed into medoid_indices).
    groups        : (n,) group labels, 'A' or 'B'.

    Returns
    -------
    cost_A : float – average nearest-medoid distance for Group A.
    cost_B : float – average nearest-medoid distance for Group B.
    """
    n = D.shape[0]
    assigned_medoids = medoid_indices[labels]            # shape (n,)
    point_costs = D[np.arange(n), assigned_medoids]     # shape (n,)

    mask_A = (groups == 'A')
    mask_B = (groups == 'B')

    cost_A = float(point_costs[mask_A].mean()) if mask_A.any() else 0.0
    cost_B = float(point_costs[mask_B].mean()) if mask_B.any() else 0.0

    return cost_A, cost_B


def calculate_fair_objective(
    D: np.ndarray,
    medoid_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    F(M) = max(GA(M), GB(M)).

    Parameters
    ----------
    D, medoid_indices, labels, groups – as above.

    Returns
    -------
    fair_obj : float
    """
    cost_A, cost_B = calculate_group_costs(D, medoid_indices, labels, groups)
    return max(cost_A, cost_B)


def find_best_fair_swap(
    D: np.ndarray,
    medoid_indices: np.ndarray,
    groups: np.ndarray,
) -> tuple:
    """
    Exhaustive one-medoid swap search using the fair objective F(M).

    For every current medoid m_j and every non-medoid candidate c,
    evaluate F(M - {m_j} + {c}) and record the best strictly-improving swap.

    Parameters
    ----------
    D             : (n, n) distance matrix.
    medoid_indices: (k,) current medoid indices.
    groups        : (n,) group labels.

    Returns
    -------
    best_fair_obj       : float  – best fair objective found.
    best_medoid_indices : (k,) np.ndarray – updated medoid set.
    improved            : bool  – True if a strictly improving swap was found.
    """
    n = D.shape[0]
    k = len(medoid_indices)
    medoid_set = set(medoid_indices.tolist())
    non_medoids = np.array([i for i in range(n) if i not in medoid_set])

    # Baseline fair objective
    labels_curr = assign_clusters(D, medoid_indices)
    best_fair_obj = calculate_fair_objective(D, medoid_indices, labels_curr, groups)
    best_medoid_indices = medoid_indices.copy()
    improved = False

    for j in range(k):
        for c in non_medoids:
            candidate = medoid_indices.copy()
            candidate[j] = c
            labels_cand = assign_clusters(D, candidate)
            fair_obj = calculate_fair_objective(D, candidate, labels_cand, groups)
            if fair_obj < best_fair_obj - 1e-10:
                best_fair_obj = fair_obj
                best_medoid_indices = candidate.copy()
                improved = True

    return best_fair_obj, best_medoid_indices, improved


def run_fair_pam(
    X: np.ndarray,
    groups: np.ndarray,
    k: int = 2,
    n_init: int = 10,
    max_iter: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run Fair-PAM with multiple random initialisations and return the best solution
    according to the fair objective F(M) = max(GA(M), GB(M)).

    Parameters
    ----------
    X        : (n, 2) feature matrix.
    groups   : (n,) group labels 'A' or 'B'.
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
        'total_cost'     : float – total distance (for comparison only).
        'cost_A'         : float – Group A average distance.
        'cost_B'         : float – Group B average distance.
        'fair_obj'       : float – F(M) = max(cost_A, cost_B).
        'n_swaps'        : int – number of accepted swaps in winning run.
        'runtime_s'      : float – wall-clock seconds for winning run.
    """
    n = X.shape[0]
    D = _euclidean_distance_matrix(X)
    # Use the SAME seed as normal PAM so both algorithms get identical init candidates
    rng = np.random.default_rng(seed)

    best_overall_fair = np.inf
    best_overall_result = None

    for init_idx in range(n_init):
        t0 = time.time()
        medoid_indices = initialize_medoids(n, k, rng)
        n_swaps = 0

        for _ in range(max_iter):
            _, new_medoids, improved = find_best_fair_swap(D, medoid_indices, groups)
            if not improved:
                break
            medoid_indices = new_medoids
            n_swaps += 1

        labels = assign_clusters(D, medoid_indices)
        cost_A, cost_B = calculate_group_costs(D, medoid_indices, labels, groups)
        fair_obj = max(cost_A, cost_B)

        # Total cost (reported for comparison, NOT used for swap selection)
        from normal_k_medoids import calculate_total_cost
        total_cost = calculate_total_cost(D, medoid_indices, labels)

        runtime = time.time() - t0

        if fair_obj < best_overall_fair:
            best_overall_fair = fair_obj
            best_overall_result = {
                'medoid_indices': medoid_indices.copy(),
                'medoid_coords':  X[medoid_indices].copy(),
                'labels':         labels.copy(),
                'total_cost':     total_cost,
                'cost_A':         cost_A,
                'cost_B':         cost_B,
                'fair_obj':       fair_obj,
                'n_swaps':        n_swaps,
                'runtime_s':      runtime,
            }

    return best_overall_result
