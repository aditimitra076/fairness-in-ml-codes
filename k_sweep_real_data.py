import numpy as np
import matplotlib.pyplot as plt

from real_data import make_titanic_fair_kmeans_data
from standard_kmeans import kmeans_multistart, clustering_cost as std_cost
from FairKmeans import fair_lloyd_multistart


def sweep_k(X, groups, k_values, n_init=10, seed=0):
    """Run both algorithms for every k in k_values.

    Returns a dict of lists, one entry per k:
        lloyd_female, lloyd_male, fair_female, fair_male
    """
    results = {
        "k": [],
        "lloyd_female": [],
        "lloyd_male": [],
        "fair_female": [],
        "fair_male": [],
    }

    maskF = groups == "A"   # female
    maskM = groups == "B"   # male

    for k in k_values:
        print(f"--- k = {k} ---")

        # ---- Standard Lloyd's ----
        _, centers, labels, n_iter = kmeans_multistart(X, k, n_init=n_init, seed=seed)
        cF = std_cost(X, labels, centers, mask=maskF)
        cM = std_cost(X, labels, centers, mask=maskM)
        print(f"  Lloyd:      female={cF:.3f}  male={cM:.3f}  (iters={n_iter})")

        # ---- Fair-Lloyd ----
        phi, costA, costB, fcenters, flabels, fn_iter = fair_lloyd_multistart(
            X, groups, k, n_init=n_init, seed=seed
        )
        print(f"  Fair-Lloyd: female={costA:.3f}  male={costB:.3f}  (iters={fn_iter})")

        results["k"].append(k)
        results["lloyd_female"].append(cF)
        results["lloyd_male"].append(cM)
        results["fair_female"].append(costA)
        results["fair_male"].append(costB)

    return results


def plot_sweep(results, save_path=None, title="Titanic dataset"):
    k = results["k"]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(k, results["lloyd_female"], color="#a6216b", marker="o",
            markersize=4, label="Lloyd (Female)")
    ax.plot(k, results["lloyd_male"], color="#1a1a80", marker="o",
            markersize=4, label="Lloyd (Male)")

    # Fair-Lloyd: female/male costs should be (near) identical -> a single
    # "Equal Cost" line, averaging the two just in case they differ by a
    # hair numerically (matches how the paper draws the Adult-dataset panel).
    fair_equal = [(f + m) / 2 for f, m in zip(results["fair_female"], results["fair_male"])]
    ax.plot(k, fair_equal, color="#e0a52c", marker="o",
            markersize=4, label="Fair-Lloyd (Equal Cost)")

    ax.set_xlabel("k")
    ax.set_ylabel("cost")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    X, groups, feature_cols = make_titanic_fair_kmeans_data()
    print(f"Loaded Titanic data: {X.shape[0]} passengers, features={feature_cols}")

    k_values = list(range(4, 16))
    results = sweep_k(X, groups, k_values, n_init=10, seed=0)
    plot_sweep(results, save_path="titanic_fairness_sweep.png")
