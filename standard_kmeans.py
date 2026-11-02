

import numpy as np
from Data import make_fair_kmeans_demo_data


def kmeans_lloyd(X, k, init_centers, max_iter=300, tol=1e-8):
    
    centers = np.array(init_centers, dtype=float).copy()

#counts iteration
    for it in range(1, max_iter + 1):
        # ---- Step 1: assign each point to nearest center ----
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # (n, k)
        labels = np.argmin(dists, axis=1)

        # ---- Step 2: recompute centers as cluster means ----
        new_centers = centers.copy()
        for i in range(k):
            members = X[labels == i]
            if len(members) > 0:
                #getting the mean
                new_centers[i] = members.mean(axis=0)
            # if a cluster becomes empty, keep its old center

        #check for convergence
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break

    return centers, labels, it


#not part of algo just reportng results afterwards
def clustering_cost(X, labels, centers, mask=None):
    
    if mask is not None:
        X, labels = X[mask], labels[mask]
    if len(X) == 0:
        return 0.0
    d2 = np.sum((X - centers[labels]) ** 2, axis=1)
    return d2.mean()


def kmeans_multistart(X, k, n_init=30, seed=0, max_iter=300):
    """Standard practice for k-means: try several random initializations and
    keep the one with the lowest overall k-means cost (inertia)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        init = X[rng.choice(len(X), size=k, replace=False)]
        centers, labels, n_iter = kmeans_lloyd(X, k, init, max_iter=max_iter)
        cost = clustering_cost(X, labels, centers)
        if best is None or cost < best[0]:
            best = (cost, centers.copy(), labels.copy(), n_iter)
    return best  # (overall_cost, centers, labels, n_iter)

8
def run_standard_kmeans_demo(seed=0, k=2, plot=True, save_path=None, n_init=30):
    X, groups = make_fair_kmeans_demo_data(seed=seed)

    _, centers, labels, n_iter = kmeans_multistart(X, k, n_init=n_init, seed=seed)

    costA = clustering_cost(X, labels, centers, mask=(groups == 'A'))
    costB = clustering_cost(X, labels, centers, mask=(groups == 'B'))
    overall = clustering_cost(X, labels, centers)

    print(f"Standard k-means: best of {n_init} initializations, converged in {n_iter} iterations")
    print(f"Centers:\n{centers}")
    print(f"Average cost - Group A (blue):   {costA:.2f}")
    print(f"Average cost - Group B (purple): {costB:.2f}")
    print(f"Overall average cost:            {overall:.2f}")
    print(f"Bias (max/min group cost ratio): {max(costA, costB) / max(min(costA, costB), 1e-9):.2f}x")

    if plot:
        plot_result(X, groups, labels, centers, title="Standard k-means",
                    save_path=save_path)

    return centers, labels, X, groups


#plotting
def plot_result(X, groups, labels, centers, title, save_path=None):
    import matplotlib.pyplot as plt

#just doing teh figure dimensions, colors and display things
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    colors = {'A': '#a8c3e0', 'B': '#a04ea0'}  # blue-ish, purple-ish
    for g, c in colors.items():
        mask = groups == g
        ax.scatter(X[mask, 0], X[mask, 1], s=25, facecolors='none',
                   edgecolors=c, linewidths=1.2, label=f"Group {g}")

    ax.scatter(centers[:, 0], centers[:, 1], s=90, c='navy', marker='o',
               edgecolors='black', zorder=5, label="centers")

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(-40, 140)
    ax.set_ylim(-20, 160)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    run_standard_kmeans_demo(save_path="standard_kmeans_result.png")