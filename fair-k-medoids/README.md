# Fair k-Medoids: Normal PAM vs Fair-PAM

A from-scratch implementation comparing standard **Partitioning Around Medoids (PAM)**
with a **fairness-aware PAM heuristic (Fair-PAM)** on a two-group synthetic dataset.

Inspired by Figure 2 in  
> Ghadiri, M., Samadi, S., & Vempala, S. (2021).  
> *Socially Fair k-Means Clustering*. FAccT 2021.

---

## Table of Contents

1. [What is k-Medoids?](#what-is-k-medoids)
2. [What is a Medoid?](#what-is-a-medoid)
3. [How Normal PAM Works](#how-normal-pam-works)
4. [How Fair-PAM Works](#how-fair-pam-works)
5. [Objective Difference](#objective-difference)
6. [Synthetic Dataset](#synthetic-dataset)
7. [Why Group A is Majority and Group B is Minority](#group-sizes)
8. [Why k = 2](#why-k--2)
9. [Expected Partition Patterns](#expected-partition-patterns)
10. [Limitations](#limitations)
11. [Running the Experiment](#running-the-experiment)
12. [Output Files](#output-files)

---

## What is k-Medoids?

**k-Medoids** is a centroid-based clustering algorithm that, unlike k-Means, requires
cluster centres to be **actual data points** (called *medoids*) rather than arithmetic
means.  This makes it more robust to outliers and naturally applicable to datasets where
a mean coordinate may not correspond to any real observation.

---

## What is a Medoid?

A **medoid** of a cluster is the data point within that cluster that minimises the total
(or average) distance to all other points in the cluster.  In PAM's swap step the medoid
is any data point that, when chosen as the cluster representative, minimises the objective.

---

## How Normal PAM Works

**PAM** (Partitioning Around Medoids) is the classic k-Medoids algorithm.

### Objective

$$J(\mathcal{M}) = \sum_{x \in X} \min_{m \in \mathcal{M}} \|x - m\|_2$$

The goal is to find the set of `k` medoids $\mathcal{M}$ that minimises the **total
nearest-medoid distance** across all observations.

### Algorithm

1. **Initialise** — sample `k` distinct data points as initial medoids.
2. **Assign** — assign every observation to its nearest medoid (Euclidean distance).
3. **Swap search** — for every current medoid $m_j$ and every non-medoid candidate $c$,
   evaluate $J(\mathcal{M} - \{m_j\} + \{c\})$.
4. **Accept** — apply the swap that gives the greatest strictly-decreasing change in $J$.
5. **Repeat** — iterate until no improving swap exists or the iteration cap is reached.
6. **Multi-start** — run from `n_init` random starting medoid pairs; keep the best.

---

## How Fair-PAM Works

**Fair-PAM** is a fairness-aware PAM heuristic explored in this project.  It changes
only the *swap-selection objective* while keeping the assignment rule identical (nearest
medoid).

### Per-group average costs

$$G_A(\mathcal{M}) = \frac{1}{|A|} \sum_{x \in A} \min_{m \in \mathcal{M}} \|x - m\|_2$$

$$G_B(\mathcal{M}) = \frac{1}{|B|} \sum_{x \in B} \min_{m \in \mathcal{M}} \|x - m\|_2$$

### Fair objective

$$F(\mathcal{M}) = \max\bigl(G_A(\mathcal{M}),\; G_B(\mathcal{M})\bigr)$$

### Algorithm

Same structure as PAM, but in step 3–4 the algorithm selects the swap that most
reduces $F(\mathcal{M})$ (not $J(\mathcal{M})$).  A swap is accepted only if it
**strictly** lowers $F$.

Group labels are used exclusively to compute $G_A$ and $G_B$; they are never used as
clustering features.

---

## Objective Difference

| Property | Normal PAM | Fair-PAM |
|---|---|---|
| Swap criterion | Minimise total distance $J$ | Minimise max group-avg distance $F$ |
| Assignment rule | Nearest medoid | Nearest medoid (identical) |
| Fairness awareness | None | Explicit via $\max(G_A, G_B)$ |
| Worst-case group cost | Not controlled | Minimised |

---

## Synthetic Dataset

The dataset contains **500 observations** with two real-valued features (X, Y):

| Sub-population | Group | Points | Centre | Std |
|---|---|---|---|---|
| blueL  | A | 200 | (0, 115)  | (15, 12) |
| blueR  | A | 200 | (90, 115) | (15, 12) |
| purpleL | B | 50  | (0, 20)   | (12, 10) |
| purpleR | B | 50  | (90, 20)  | (12, 10) |

Generated with `numpy.random.default_rng(seed=0)`.

The dataset has two distinct spatial structures:

- **Horizontal** (left/right): blobs share the same X ≈ 0 or X ≈ 90, regardless of group.
- **Vertical** (top/bottom): Group A sits at Y ≈ 115 while Group B sits at Y ≈ 20.

---

## Group Sizes

**Group A** (blue) has 400 points; **Group B** (purple) has 100 points.  
This 4:1 imbalance means a total-distance minimiser is biased toward solutions
that serve the majority group well, potentially at the expense of the minority.

The asymmetry deliberately mirrors real-world settings where demographic groups
differ substantially in size.

---

## Why k = 2

Two clusters match the two dominant structures in the dataset (left vs right, or
top vs bottom).  Using `k = 2` makes the trade-off between the two possible
partitions clear and interpretable: the two algorithms are expected to choose
different partitions for different reasons.

---

## Expected Partition Patterns

### Normal PAM — expected left/right partition

Because the horizontal blobs (blueL + purpleL) and (blueR + purpleR) are each
relatively compact in Euclidean space, a medoid at roughly (0, ~90) and one at
roughly (90, ~90) minimises *total* distance by grouping points with similar X
coordinates.  The majority group (Group A) drives the solution because it contains
4× as many points.

### Fair-PAM — expected top/bottom partition

Placing one medoid near (≈0, 115) or (≈90, 115) (Group A territory) and another
near (≈0, 20) or (≈90, 20) (Group B territory) yields comparable average distances
for both groups, because each group is close to its representative.  This minimises
$F = \max(G_A, G_B)$, even if total cost is higher.

These are **expected** patterns based on the geometry of the dataset; they are not
hard-coded.  If the optimisation produces a different result, the actual outcome is
reported and the objective values explain why.

---

## Limitations

1. **Local search** — PAM and Fair-PAM are local swap heuristics.  They can converge
   to a local optimum rather than the global minimiser.  Multiple random initialisations
   mitigate but do not eliminate this risk.

2. **Single fairness metric** — minimising $\max(G_A, G_B)$ improves one specific notion
   of fairness (equalising worst-group average cost).  It does not guarantee fairness
   under other definitions (e.g., equal representation in each cluster, proportional
   coverage, individual fairness).

3. **Binary groups** — the current implementation handles two groups.  Extension to
   multiple groups would require replacing $\max(G_A, G_B)$ with $\max_{g} G_g$.

4. **Scalability** — exhaustive one-medoid swap search is $O(k \cdot (n-k))$ per
   iteration.  For 500 points this is fast; for large datasets a restricted candidate
   set (e.g., CLARA, FastPAM) would be needed.

5. **No novelty claim** — Fair-PAM is described here as a fairness-aware PAM
   heuristic explored for research purposes.  It is not presented as a proven novel
   algorithm, and its theoretical guarantees have not been established.

---

## Running the Experiment

```bash
# (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the experiment
python main.py
```

Expected runtime: a few minutes on a modern laptop (exhaustive swap search over 500 points).

---

## Output Files

| File | Description |
|---|---|
| `results/normal_vs_fair_k_medoids.png` | Side-by-side comparison figure |
| `results/standard_k_medoids.png` | Normal PAM standalone figure |
| `results/fair_k_medoids.png` | Fair-PAM standalone figure |
| `results/metrics.csv` | Per-algorithm numerical metrics |
| `results/metrics.json` | Same metrics + comparison, JSON format |
| `results/assignments.csv` | Per-observation cluster assignments |
| `results/cluster_group_distribution.csv` | Group counts per cluster per algorithm |
