"""
data.py
-------
Synthetic dataset generator for the Fair k-Medoids experiment.

The dataset mimics the two-group, two-cluster layout used in
"Socially Fair k-Means Clustering" (Ghadiri, Samadi, Vempala, 2021),
Figure 2, adapted for k-Medoids.

Group structure
---------------
  Group A (majority, blue):  400 points
    - blueL  : blob centred at  (0, 115), sigma (15, 12)
    - blueR  : blob centred at (90, 115), sigma (15, 12)

  Group B (minority, purple): 100 points
    - purpleL: blob centred at  (0,  20), sigma (12, 10)
    - purpleR: blob centred at (90,  20), sigma (12, 10)

Total: 500 observations, 2 features (X, Y).
"""

import numpy as np


def make_fair_kmeans_demo_data(seed: int = 0):
    """
    Generate the two-group, four-blob synthetic dataset.

    Parameters
    ----------
    seed : int
        NumPy random seed for reproducibility (default 0).

    Returns
    -------
    X : np.ndarray, shape (500, 2)
        Feature matrix (X coordinate, Y coordinate).
    groups : np.ndarray, shape (500,), dtype str
        Group label for each observation: 'A' (majority) or 'B' (minority).
    """
    rng = np.random.default_rng(seed)

    # ---- Group A : majority group, two upper blobs ----
    nA = 200

    blueL = rng.normal(
        loc=(0, 115),
        scale=(15, 12),
        size=(nA, 2)
    )

    blueR = rng.normal(
        loc=(90, 115),
        scale=(15, 12),
        size=(nA, 2)
    )

    # ---- Group B : minority group, two lower blobs ----
    nB = 50

    purpleL = rng.normal(
        loc=(0, 20),
        scale=(12, 10),
        size=(nB, 2)
    )

    purpleR = rng.normal(
        loc=(90, 20),
        scale=(12, 10),
        size=(nB, 2)
    )

    X = np.vstack([blueL, blueR, purpleL, purpleR])

    groups = np.array(
        ['A'] * (nA * 2) + ['B'] * (nB * 2)
    )

    return X, groups


if __name__ == "__main__":
    X, groups = make_fair_kmeans_demo_data(seed=0)
    print(f"Dataset shape : {X.shape}")
    print(f"Group A count : {(groups == 'A').sum()}")
    print(f"Group B count : {(groups == 'B').sum()}")
    print(f"X range       : [{X[:, 0].min():.1f}, {X[:, 0].max():.1f}]")
    print(f"Y range       : [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}]")
