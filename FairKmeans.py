

import numpy as np
from Data import make_fair_kmeans_demo_data


#line search
def line_search(X, groups, labels, k, T=60, eps=1e-12):
   
    d = X.shape[1]
    maskA = groups == 'A'
    maskB = groups == 'B'
    nA, nB = maskA.sum(), maskB.sum()

    muA = np.zeros((k, d))
    muB = np.zeros((k, d))
    alpha = np.zeros(k)
    beta = np.zeros(k)
    l = np.zeros(k)

    base_A = 0.0  # Delta(M^A, U ∩ A) / |A|
    base_B = 0.0  # Delta(M^B, U ∩ B) / |B|

#cluster loop
    for i in range(k):
        Ai = X[maskA & (labels == i)]
        Bi = X[maskB & (labels == i)]
        alpha[i] = len(Ai) / nA if nA > 0 else 0.0
        beta[i] = len(Bi) / nB if nB > 0 else 0.0

        muA[i] = Ai.mean(axis=0) if len(Ai) > 0 else 0.0
        muB[i] = Bi.mean(axis=0) if len(Bi) > 0 else 0.0
        l[i] = np.linalg.norm(muA[i] - muB[i]) if (len(Ai) > 0 and len(Bi) > 0) else 0.0

        if len(Ai) > 0:
            base_A += np.sum((Ai - muA[i]) ** 2) / nA
        if len(Bi) > 0:
            base_B += np.sum((Bi - muB[i]) ** 2) / nB

    def f_A(x):
        return base_A + np.sum(alpha * x ** 2)

    def f_B(x):
        return base_B + np.sum(beta * (l - x) ** 2)

#lemma 2 fro paper: 
    def x_of_gamma(gamma):
        denom = gamma * alpha + (1 - gamma) * beta
        x = np.where(denom > eps, (1 - gamma) * beta * l / np.maximum(denom, eps), 0.0)
        return np.clip(x, 0.0, l)

    # ---- Algorithm 2: bisection on gamma in [0, 1] ----
    gamma = 0.5
    step = 0.25
    x = x_of_gamma(gamma)
    for t in range(1, T + 1):
        x = x_of_gamma(gamma)
        fa, fb = f_A(x), f_B(x)
        if fa > fb:
            gamma += step
        elif fa < fb:
            gamma -= step
        else:
            break
        step *= 0.5
    x = x_of_gamma(gamma)

    # ---- Recover centers from x_i on the segment [mu^A_i, mu^B_i] ----
    centers = np.zeros((k, d))
    for i in range(k):
        if l[i] > eps:
            centers[i] = muA[i] + (x[i] / l[i]) * (muB[i] - muA[i])
        elif alpha[i] > 0:
            centers[i] = muA[i]           # only group A present in this cluster
        elif beta[i] > 0:
            centers[i] = muB[i]           # only group B present in this cluster
        # if cluster empty, leave center at 0 (caller should handle / keep old)

    return centers


# ----------------------------------------------------------------------
# Algorithm 1: Fair-Lloyd
# ----------------------------------------------------------------------
def fair_lloyd(X, groups, k, init_centers, max_iter=100, tol=1e-8, ls_iters=60):
    centers = np.array(init_centers, dtype=float).copy()
    labels = None

    for it in range(1, max_iter + 1):
        # ---- Step 1: assign each point to its nearest current center ----
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)

        if labels is not None and np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

        # ---- Step 2: fair centers for this partition, via line search ----
        new_centers = line_search(X, groups, labels, k, T=ls_iters)
        # keep old center for any (rare) empty cluster
        empty = np.all(new_centers == 0, axis=1) & np.array(
            [(labels == i).sum() == 0 for i in range(k)]
        )
        new_centers[empty] = centers[empty]

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break

    return centers, labels, it


#just the clustering cost
def clustering_cost(X, labels, centers, mask=None):
    if mask is not None:
        X, labels = X[mask], labels[mask]
    if len(X) == 0:
        return 0.0
    d2 = np.sum((X - centers[labels]) ** 2, axis=1)
    return d2.mean()


def fair_lloyd_multistart(X, groups, k, n_init=30, seed=0, max_iter=100, ls_iters=60):
    
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        init = X[rng.choice(len(X), size=k, replace=False)]
        centers, labels, n_iter = fair_lloyd(X, groups, k, init,
                                             max_iter=max_iter, ls_iters=ls_iters)
        costA = clustering_cost(X, labels, centers, mask=(groups == 'A'))
        costB = clustering_cost(X, labels, centers, mask=(groups == 'B'))
        phi = max(costA, costB)
        if best is None or phi < best[0]:
            best = (phi, costA, costB, centers.copy(), labels.copy(), n_iter)
    return best  # (phi, costA, costB, centers, labels, n_iter)


#plot
def run_fair_lloyd_demo(seed=0, k=2, plot=True, save_path=None, n_init=30):
    X, groups = make_fair_kmeans_demo_data(seed=seed)

    phi, costA, costB, centers, labels, n_iter = fair_lloyd_multistart(
        X, groups, k, n_init=n_init, seed=seed
    )
    overall = clustering_cost(X, labels, centers)

    print(f"Fair-Lloyd: best of {n_init} initializations, best run converged in {n_iter} iterations")
    print(f"Centers:\n{centers}")
    print(f"Average cost - Group A (blue):   {costA:.2f}")
    print(f"Average cost - Group B (purple): {costB:.2f}")
    print(f"Overall average cost:            {overall:.2f}")
    print(f"Bias (max/min group cost ratio): {max(costA, costB) / max(min(costA, costB), 1e-9):.2f}x")

    if plot:
        from standard_kmeans import plot_result
        plot_result(X, groups, labels, centers, title="Fair k-means",
                    save_path=save_path)

    return centers, labels, X, groups


if __name__ == "__main__":
    run_fair_lloyd_demo(save_path="fair_kmeans_result.png")