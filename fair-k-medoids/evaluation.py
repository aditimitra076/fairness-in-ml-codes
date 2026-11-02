"""
evaluation.py
-------------
Numerical evaluation for Normal PAM and Fair-PAM results.

Metrics computed
----------------
  - Total distance (sum of nearest-medoid distances).
  - Group A average distance.
  - Group B average distance.
  - Maximum group-average cost (the fair objective).
  - Absolute group-cost gap |GA - GB|.
  - Number of accepted swaps.
  - Optimisation runtime (seconds).
  - Percentage improvement in worst-group cost (Fair vs Normal).
  - Percentage change in total cost (Fair vs Normal).

Also computes per-cluster group-membership counts.
"""

import json
import csv
import os
import numpy as np

from normal_k_medoids import (
    _euclidean_distance_matrix,
    assign_clusters,
    calculate_total_cost,
)
from fair_k_medoids import calculate_group_costs


# ---------------------------------------------------------------------------
# Single-solution evaluation
# ---------------------------------------------------------------------------

def evaluate_solution(
    X: np.ndarray,
    groups: np.ndarray,
    result: dict,
    algorithm_name: str,
) -> dict:
    """
    Compute a full set of metrics for one algorithm's result.

    Parameters
    ----------
    X              : (n, 2) feature matrix.
    groups         : (n,) group labels 'A' / 'B'.
    result         : dict returned by run_normal_pam() or run_fair_pam().
    algorithm_name : str label for the algorithm.

    Returns
    -------
    metrics : dict
    """
    D = _euclidean_distance_matrix(X)
    medoid_indices = result['medoid_indices']
    labels = result['labels']

    cost_A, cost_B = calculate_group_costs(D, medoid_indices, labels, groups)
    total_cost = calculate_total_cost(D, medoid_indices, labels)
    fair_obj = max(cost_A, cost_B)
    gap = abs(cost_A - cost_B)

    metrics = {
        'algorithm':       algorithm_name,
        'total_cost':      round(total_cost, 4),
        'cost_A':          round(cost_A, 4),
        'cost_B':          round(cost_B, 4),
        'fair_obj':        round(fair_obj, 4),
        'group_cost_gap':  round(gap, 4),
        'n_swaps':         result['n_swaps'],
        'runtime_s':       round(result['runtime_s'], 4),
        'medoid_indices':  medoid_indices.tolist(),
        'medoid_coords':   result['medoid_coords'].tolist(),
    }
    return metrics


# ---------------------------------------------------------------------------
# Cross-algorithm comparison
# ---------------------------------------------------------------------------

def compare_algorithms(metrics_normal: dict, metrics_fair: dict) -> dict:
    """
    Compute relative improvements of Fair-PAM over Normal PAM.

    Returns
    -------
    comparison : dict
    """
    worst_normal = metrics_normal['fair_obj']
    worst_fair   = metrics_fair['fair_obj']
    total_normal = metrics_normal['total_cost']
    total_fair   = metrics_fair['total_cost']

    pct_worst_group_improvement = (
        (worst_normal - worst_fair) / worst_normal * 100
        if worst_normal > 0 else 0.0
    )
    pct_total_cost_change = (
        (total_fair - total_normal) / total_normal * 100
        if total_normal > 0 else 0.0
    )

    return {
        'pct_worst_group_improvement': round(pct_worst_group_improvement, 2),
        'pct_total_cost_change':        round(pct_total_cost_change, 2),
        'normal_fair_obj':              metrics_normal['fair_obj'],
        'fair_pam_fair_obj':            metrics_fair['fair_obj'],
        'normal_total_cost':            metrics_normal['total_cost'],
        'fair_pam_total_cost':          metrics_fair['total_cost'],
    }


# ---------------------------------------------------------------------------
# Cluster group distribution
# ---------------------------------------------------------------------------

def calculate_group_distribution(
    labels: np.ndarray,
    groups: np.ndarray,
    algorithm_name: str,
    k: int = 2,
) -> list:
    """
    For each cluster, count Group A and Group B members.

    Returns
    -------
    rows : list of dicts, one per cluster.
    """
    rows = []
    for cluster_id in range(k):
        mask = (labels == cluster_id)
        count_A = int((groups[mask] == 'A').sum())
        count_B = int((groups[mask] == 'B').sum())
        rows.append({
            'Algorithm':     algorithm_name,
            'Cluster':       cluster_id + 1,
            'Group_A_count': count_A,
            'Group_B_count': count_B,
        })
    return rows


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_metrics(
    metrics_normal: dict,
    metrics_fair:   dict,
    comparison:     dict,
    out_dir: str = 'results',
) -> None:
    """
    Save metrics_normal, metrics_fair, and comparison to CSV and JSON.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- JSON ----
    payload = {
        'normal_pam': metrics_normal,
        'fair_pam':   metrics_fair,
        'comparison': comparison,
    }
    json_path = os.path.join(out_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    # ---- CSV ----
    csv_path = os.path.join(out_dir, 'metrics.csv')
    fieldnames = [
        'algorithm', 'total_cost', 'cost_A', 'cost_B',
        'fair_obj', 'group_cost_gap', 'n_swaps', 'runtime_s',
        'medoid_indices', 'medoid_coords',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in [metrics_normal, metrics_fair]:
            row = {k: m[k] for k in fieldnames}
            writer.writerow(row)

    print(f"  Metrics saved -> {json_path}")
    print(f"  Metrics saved -> {csv_path}")


def save_assignments(
    X: np.ndarray,
    groups: np.ndarray,
    result_normal: dict,
    result_fair:   dict,
    out_dir: str = 'results',
) -> None:
    """
    Save per-observation cluster assignments for both algorithms.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'assignments.csv')
    n = X.shape[0]

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'obs_idx', 'X', 'Y', 'group',
            'normal_pam_cluster', 'normal_pam_medoid_idx',
            'fair_pam_cluster',   'fair_pam_medoid_idx',
        ])
        for i in range(n):
            nl = result_normal['labels'][i]
            fl = result_fair['labels'][i]
            writer.writerow([
                i,
                round(float(X[i, 0]), 4),
                round(float(X[i, 1]), 4),
                groups[i],
                nl + 1,
                int(result_normal['medoid_indices'][nl]),
                fl + 1,
                int(result_fair['medoid_indices'][fl]),
            ])

    print(f"  Assignments saved -> {path}")


def save_group_distribution(
    rows: list,
    out_dir: str = 'results',
) -> None:
    """Save cluster group distribution table to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'cluster_group_distribution.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['Algorithm', 'Cluster', 'Group_A_count', 'Group_B_count']
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Group distribution saved -> {path}")
