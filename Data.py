"""
Synthetic dataset generator that mimics Figure 2 of
"Socially Fair k-Means Clustering" (Ghadiri, Samadi, Vempala).

Two demographic groups:
    Group A ("blue")   - majority group,  split into a left blob and a right blob
    Group B ("purple") - minority group,  split into a left blob and a right blob

Because group A has many more points, standard Lloyd's k-means is dominated by
group A's geometry and ends up splitting the data LEFT/RIGHT (i.e. it pairs each
blue blob with the purple blob that happens to share its x-range), which is bad
for group B (purple) since the resulting centers sit far above the purple points.

Fair-Lloyd instead balances the *per-group average cost*, which for this data
means splitting the data TOP/BOTTOM (all of blue in one cluster, all of purple
in the other), giving both groups a similar, low average cost.
"""

import numpy as np


def make_fair_kmeans_demo_data(seed: int = 0):
    rng = np.random.default_rng(seed)

    # ---- Group A: "blue", majority group, two blobs high up (y ~ 80-140) ----
    nA = 200
    blueL = rng.normal(loc=(0, 115), scale=(15, 12), size=(nA, 2))
    blueR = rng.normal(loc=(90, 115), scale=(15, 12), size=(nA, 2))

    # ---- Group B: "purple", minority group, two blobs down low (y ~ 0-35) ----
    # Same x-centers as blue (0 / 90) so that, when Fair-Lloyd merges each
    # group into a single top/bottom cluster, both groups have a similar
    # "half-separation" contribution to their cost -> nearly equal fair cost,
    # matching the paper's Figure 2 (right).
    nB = 50
    purpleL = rng.normal(loc=(0, 20), scale=(12, 10), size=(nB, 2))
    purpleR = rng.normal(loc=(90, 20), scale=(12, 10), size=(nB, 2))

    X = np.vstack([blueL, blueR, purpleL, purpleR])
    # group label: 'A' = blue (majority), 'B' = purple (minority)
    groups = np.array(
        ['A'] * (nA * 2) + ['B'] * (nB * 2)
    )
    return X, groups


if __name__ == "__main__":
    X, groups = make_fair_kmeans_demo_data()
    print("Total points:", len(X))
    print("Group A (blue):", (groups == 'A').sum())
    print("Group B (purple):", (groups == 'B').sum())