"""
main.py
-------
Entry point for the Fair k-Medoids experiment.

Runs Normal PAM and Fair-PAM on the same synthetic dataset,
evaluates both, generates visualisation figures, and saves all results.

Usage
-----
    python main.py

All output is written to the results/ subdirectory.
"""

import os
import sys
import time

# ---- Project modules ----
from data import make_fair_kmeans_demo_data
from normal_k_medoids import run_normal_pam
from fair_k_medoids import run_fair_pam
from evaluation import (
    evaluate_solution,
    compare_algorithms,
    calculate_group_distribution,
    save_metrics,
    save_assignments,
    save_group_distribution,
)
from visualization import (
    plot_normal_k_medoids,
    plot_fair_k_medoids,
    plot_comparison,
)

# ---- Configuration ----
K          = 2
N_INIT     = 10
MAX_ITER   = 100
DATA_SEED  = 0    # reproducible dataset
INIT_SEED  = 42   # reproducible initialisation (same for both algorithms)
RESULTS    = 'results'


def print_section(title: str) -> None:
    print('\n' + '=' * 60)
    print(f'  {title}')
    print('=' * 60)


def main():
    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    print_section('1. Generating synthetic dataset')
    X, groups = make_fair_kmeans_demo_data(seed=DATA_SEED)
    n_A = (groups == 'A').sum()
    n_B = (groups == 'B').sum()
    print(f'  Total observations : {len(X)}')
    print(f'  Group A (majority) : {n_A}')
    print(f'  Group B (minority) : {n_B}')
    print(f'  Features           : X (col 0), Y (col 1)')

    # ------------------------------------------------------------------
    # 2. Normal PAM
    # ------------------------------------------------------------------
    print_section('2. Running Normal k-Medoids (PAM)')
    print(f'  k = {K}, n_init = {N_INIT}, max_iter = {MAX_ITER}, seed = {INIT_SEED}')
    t0 = time.time()
    result_normal = run_normal_pam(
        X, k=K, n_init=N_INIT, max_iter=MAX_ITER, seed=INIT_SEED
    )
    elapsed_normal = time.time() - t0
    print(f'  Done in {elapsed_normal:.2f} s')
    print(f'  Medoid indices : {result_normal["medoid_indices"].tolist()}')
    print(f'  Medoid coords  : {result_normal["medoid_coords"].tolist()}')
    print(f'  Total cost     : {result_normal["total_cost"]:.4f}')
    print(f'  Accepted swaps : {result_normal["n_swaps"]}')

    # ------------------------------------------------------------------
    # 3. Fair-PAM
    # ------------------------------------------------------------------
    print_section('3. Running Fair k-Medoids (Fair-PAM)')
    print(f'  k = {K}, n_init = {N_INIT}, max_iter = {MAX_ITER}, seed = {INIT_SEED}')
    t0 = time.time()
    result_fair = run_fair_pam(
        X, groups, k=K, n_init=N_INIT, max_iter=MAX_ITER, seed=INIT_SEED
    )
    elapsed_fair = time.time() - t0
    print(f'  Done in {elapsed_fair:.2f} s')
    print(f'  Medoid indices : {result_fair["medoid_indices"].tolist()}')
    print(f'  Medoid coords  : {result_fair["medoid_coords"].tolist()}')
    print(f'  Fair obj F(M)  : {result_fair["fair_obj"]:.4f}')
    print(f'  Cost A         : {result_fair["cost_A"]:.4f}')
    print(f'  Cost B         : {result_fair["cost_B"]:.4f}')
    print(f'  Accepted swaps : {result_fair["n_swaps"]}')

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    print_section('4. Evaluating solutions')
    metrics_normal = evaluate_solution(X, groups, result_normal, 'Normal PAM')
    metrics_fair   = evaluate_solution(X, groups, result_fair,   'Fair-PAM')
    comparison     = compare_algorithms(metrics_normal, metrics_fair)

    print(f'\n  {"Metric":<35} {"Normal PAM":>14} {"Fair-PAM":>14}')
    print(f'  {"-"*63}')
    for key in ['total_cost', 'cost_A', 'cost_B', 'fair_obj', 'group_cost_gap']:
        print(f'  {key:<35} {str(metrics_normal[key]):>14} {str(metrics_fair[key]):>14}')
    print()
    print(f'  Worst-group improvement (Fair vs Normal) : '
          f'{comparison["pct_worst_group_improvement"]:+.2f} %')
    print(f'  Total-cost change        (Fair vs Normal) : '
          f'{comparison["pct_total_cost_change"]:+.2f} %')

    # ------------------------------------------------------------------
    # 5. Save metrics and assignments
    # ------------------------------------------------------------------
    print_section('5. Saving numerical results')
    os.makedirs(RESULTS, exist_ok=True)
    save_metrics(metrics_normal, metrics_fair, comparison, out_dir=RESULTS)
    save_assignments(X, groups, result_normal, result_fair, out_dir=RESULTS)

    dist_rows = (
        calculate_group_distribution(result_normal['labels'], groups, 'Normal PAM', k=K)
        + calculate_group_distribution(result_fair['labels'],   groups, 'Fair-PAM',   k=K)
    )
    save_group_distribution(dist_rows, out_dir=RESULTS)

    # ------------------------------------------------------------------
    # 6. Visualisation
    # ------------------------------------------------------------------
    print_section('6. Generating visualisations')
    plot_normal_k_medoids(
        X, groups, result_normal,
        out_path=os.path.join(RESULTS, 'standard_k_medoids.png'), k=K,
    )
    plot_fair_k_medoids(
        X, groups, result_fair,
        out_path=os.path.join(RESULTS, 'fair_k_medoids.png'), k=K,
    )
    plot_comparison(
        X, groups, result_normal, result_fair,
        out_path=os.path.join(RESULTS, 'normal_vs_fair_k_medoids.png'), k=K,
    )

    # ------------------------------------------------------------------
    # 7. Final summary
    # ------------------------------------------------------------------
    print_section('7. Final Summary')

    print('\n  -- Normal PAM --')
    print(f'    Medoid indices  : {result_normal["medoid_indices"].tolist()}')
    print(f'    Medoid coords   : {[[round(c, 2) for c in m] for m in result_normal["medoid_coords"].tolist()]}')
    print(f'    Total cost      : {metrics_normal["total_cost"]}')
    print(f'    Cost A          : {metrics_normal["cost_A"]}')
    print(f'    Cost B          : {metrics_normal["cost_B"]}')
    print(f'    Fair obj F(M)   : {metrics_normal["fair_obj"]}')
    print(f'    Group-cost gap  : {metrics_normal["group_cost_gap"]}')

    print('\n  -- Fair-PAM --')
    print(f'    Medoid indices  : {result_fair["medoid_indices"].tolist()}')
    print(f'    Medoid coords   : {[[round(c, 2) for c in m] for m in result_fair["medoid_coords"].tolist()]}')
    print(f'    Total cost      : {metrics_fair["total_cost"]}')
    print(f'    Cost A          : {metrics_fair["cost_A"]}')
    print(f'    Cost B          : {metrics_fair["cost_B"]}')
    print(f'    Fair obj F(M)   : {metrics_fair["fair_obj"]}')
    print(f'    Group-cost gap  : {metrics_fair["group_cost_gap"]}')

    print('\n  -- Cross-algorithm comparison --')
    print(f'    Worst-group improvement : {comparison["pct_worst_group_improvement"]:+.2f} %')
    print(f'    Total-cost change       : {comparison["pct_total_cost_change"]:+.2f} %')

    # ---- Partition pattern report ----
    print('\n  -- Partition pattern --')
    nm = result_normal['medoid_coords']
    fm = result_fair['medoid_coords']
    nm_dx = abs(nm[0][0] - nm[1][0])   # horizontal separation of normal medoids
    fm_dy = abs(fm[0][1] - fm[1][1])   # vertical separation of fair medoids
    nm_dy = abs(nm[0][1] - nm[1][1])
    fm_dx = abs(fm[0][0] - fm[1][0])

    print(f'    Normal PAM medoid dX (horizontal) = {nm_dx:.1f},  dY (vertical) = {nm_dy:.1f}')
    if nm_dx > nm_dy:
        print('    -> Normal PAM shows predominantly LEFT/RIGHT partition (expected).')
    else:
        print('    -> Normal PAM shows predominantly TOP/BOTTOM partition.')

    print(f'    Fair-PAM medoid dX (horizontal)   = {fm_dx:.1f},  dY (vertical) = {fm_dy:.1f}')
    if fm_dy > fm_dx:
        print('    -> Fair-PAM shows predominantly TOP/BOTTOM partition (expected).')
    else:
        print('    -> Fair-PAM shows predominantly LEFT/RIGHT partition.')

    print('\n  -- Files created --')
    files = [
        os.path.join(RESULTS, 'normal_vs_fair_k_medoids.png'),
        os.path.join(RESULTS, 'standard_k_medoids.png'),
        os.path.join(RESULTS, 'fair_k_medoids.png'),
        os.path.join(RESULTS, 'metrics.csv'),
        os.path.join(RESULTS, 'metrics.json'),
        os.path.join(RESULTS, 'assignments.csv'),
        os.path.join(RESULTS, 'cluster_group_distribution.csv'),
    ]
    for f in files:
        status = '[OK]' if os.path.exists(f) else '[MISSING]'
        print(f'    {status}  {f}')

    print('\nDone.\n')


if __name__ == '__main__':
    main()
