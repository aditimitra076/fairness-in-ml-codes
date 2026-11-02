"""
visualization.py
----------------
Plotting functions for Normal PAM vs Fair-PAM comparison.

Visual style is inspired by Figure 2 of
"Socially Fair k-Means Clustering" (Ghadiri, Samadi, Vempala, 2021).

Colour conventions
------------------
  Group A (majority) : blue   (#4472C4)
  Group B (minority) : purple (#9B59B6)
  Medoids            : black stars (*)

Cluster boundaries are convex hulls of the assigned observations.
  Cluster 1 : solid line
  Cluster 2 : dashed line
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless backend; no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

GROUP_COLORS = {
    'A': '#4472C4',   # blue  (majority)
    'B': '#9B59B6',   # purple (minority)
}
MEDOID_COLOR  = 'black'
MEDOID_MARKER = '*'
MEDOID_SIZE   = 350

CLUSTER_LINE_STYLES = ['solid', 'dashed']
CLUSTER_LINE_COLORS = ['#2c3e50', '#7f8c8d']  # dark slate / grey

POINT_SIZE  = 18
POINT_ALPHA = 0.75
HULL_ALPHA  = 0.18
HULL_LW     = 1.6


# ---------------------------------------------------------------------------
# Convex hull helper
# ---------------------------------------------------------------------------

def _draw_hull(ax, points: np.ndarray, color: str, linestyle: str, alpha: float = HULL_ALPHA) -> None:
    """
    Draw a convex hull boundary around *points* on *ax*.
    If fewer than 3 unique points exist, draws a scatter or line instead.
    """
    if len(points) < 3:
        ax.scatter(points[:, 0], points[:, 1], s=POINT_SIZE * 3,
                   color=color, alpha=0.5, zorder=2)
        return

    try:
        hull = ConvexHull(points)
        # Close the hull polygon
        verts = np.append(hull.vertices, hull.vertices[0])
        ax.plot(
            points[verts, 0], points[verts, 1],
            color=color, linestyle=linestyle, linewidth=HULL_LW, zorder=3,
        )
        ax.fill(
            points[hull.vertices, 0], points[hull.vertices, 1],
            color=color, alpha=alpha, zorder=1,
        )
    except Exception:
        # Degenerate hull: just skip the fill
        pass


# ---------------------------------------------------------------------------
# Single-panel plot
# ---------------------------------------------------------------------------

def _plot_panel(
    ax,
    X: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    medoid_indices: np.ndarray,
    title: str,
    k: int = 2,
) -> None:
    """
    Draw one clustering panel (scatter + hulls + medoids) on *ax*.
    """
    n_clusters = k
    hull_colors = [CLUSTER_LINE_COLORS[i % len(CLUSTER_LINE_COLORS)] for i in range(n_clusters)]
    hull_styles = [CLUSTER_LINE_STYLES[i % len(CLUSTER_LINE_STYLES)] for i in range(n_clusters)]

    # ---- Convex hulls (drawn first so points appear on top) ----
    for c in range(n_clusters):
        mask_c = (labels == c)
        if mask_c.sum() > 0:
            _draw_hull(ax, X[mask_c], hull_colors[c], hull_styles[c])

    # ---- Scatter points ----
    for grp, color in GROUP_COLORS.items():
        mask = (groups == grp)
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=color, s=POINT_SIZE, alpha=POINT_ALPHA,
            edgecolors='none', zorder=4,
            label=f'Group {"A (majority)" if grp == "A" else "B (minority)"}',
        )

    # ---- Medoid stars ----
    for mi in medoid_indices:
        ax.scatter(
            X[mi, 0], X[mi, 1],
            marker=MEDOID_MARKER, s=MEDOID_SIZE,
            c=MEDOID_COLOR, zorder=6, edgecolors='white', linewidths=0.5,
        )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_normal_k_medoids(
    X: np.ndarray,
    groups: np.ndarray,
    result: dict,
    out_path: str = 'results/standard_k_medoids.png',
    k: int = 2,
) -> None:
    """Save the Normal PAM clustering figure."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    _plot_panel(ax, X, groups, result['labels'], result['medoid_indices'],
                title='Normal k-Medoids (PAM)', k=k)
    _add_legend(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_fair_k_medoids(
    X: np.ndarray,
    groups: np.ndarray,
    result: dict,
    out_path: str = 'results/fair_k_medoids.png',
    k: int = 2,
) -> None:
    """Save the Fair-PAM clustering figure."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    _plot_panel(ax, X, groups, result['labels'], result['medoid_indices'],
                title='Fair k-Medoids (Fair-PAM)', k=k)
    _add_legend(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Comparison figure (main deliverable)
# ---------------------------------------------------------------------------

def plot_comparison(
    X: np.ndarray,
    groups: np.ndarray,
    result_normal: dict,
    result_fair: dict,
    out_path: str = 'results/normal_vs_fair_k_medoids.png',
    k: int = 2,
) -> None:
    """
    Generate the side-by-side comparison figure.
    Left  : Normal k-Medoids (PAM)
    Right : Fair k-Medoids (Fair-PAM)
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor='white')
    fig.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.10, wspace=0.18)

    # Shared axis limits
    x_min, x_max = X[:, 0].min() - 20, X[:, 0].max() + 20
    y_min, y_max = X[:, 1].min() - 15, X[:, 1].max() + 15

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    _plot_panel(axes[0], X, groups,
                result_normal['labels'], result_normal['medoid_indices'],
                title='Normal k-Medoids (PAM)', k=k)

    _plot_panel(axes[1], X, groups,
                result_fair['labels'], result_fair['medoid_indices'],
                title='Fair k-Medoids (Fair-PAM)', k=k)

    # Title at the very top
    fig.suptitle(
        'Normal k-Medoids vs Fair k-Medoids  |  k = 2',
        fontsize=13, fontweight='bold', y=0.99,
    )

    # Legend just below the title, within the reserved 20% top margin
    fig.legend(
        handles=_make_legend_handles(),
        loc='upper center', ncol=5,
        fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, 0.90),
    )

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------

def _make_legend_handles():
    handles = [
        mpatches.Patch(color=GROUP_COLORS['A'], label='Group A – majority (blue)'),
        mpatches.Patch(color=GROUP_COLORS['B'], label='Group B – minority (purple)'),
        Line2D([0], [0], marker=MEDOID_MARKER, color='w', markerfacecolor=MEDOID_COLOR,
               markersize=12, label='Medoid'),
        Line2D([0], [0], color=CLUSTER_LINE_COLORS[0], linestyle='solid',
               linewidth=2, label='Cluster 1 boundary'),
        Line2D([0], [0], color=CLUSTER_LINE_COLORS[1], linestyle='dashed',
               linewidth=2, label='Cluster 2 boundary'),
    ]
    return handles


def _add_legend(fig) -> None:
    """Add a compact legend at the top of a single-panel figure."""
    fig.legend(
        handles=_make_legend_handles(),
        loc='upper center', ncol=5,
        fontsize=8, frameon=True,
        bbox_to_anchor=(0.5, 1.01),
    )


def _add_shared_legend(fig) -> None:
    """Add a shared legend at the top of the comparison figure."""
    fig.legend(
        handles=_make_legend_handles(),
        loc='upper center', ncol=5,
        fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, 1.00),
    )
