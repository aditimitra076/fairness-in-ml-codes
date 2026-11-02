
import io
import urllib.request

import numpy as np
import pandas as pd

TITANIC_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"


def load_titanic_raw():
    """Download the raw Titanic CSV (891 passengers) and return it as a
    pandas DataFrame. Requires network access to raw.githubusercontent.com."""
    req = urllib.request.Request(TITANIC_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw_bytes = urllib.request.urlopen(req, timeout=20).read()
    df = pd.read_csv(io.BytesIO(raw_bytes))
    return df


def make_titanic_fair_kmeans_data():
    """Return (X, groups) ready for clustering.

    X      : (n, d) standardized numeric feature matrix
    groups : (n,) array of 'A' (female) / 'B' (male) labels
    """
    df = load_titanic_raw()

    # Keep only rows with a known sex (there are none missing, but be safe)
    # part of data pre processijg
    df = df.dropna(subset=["sex"]).reset_index(drop=True)

   
    feature_cols = ["pclass", "age", "sibsp", "parch", "fare"] #these are the cols we cluster on

    #filling in the missing age values with median as less sensitive to outliers thatn mean
    df["age"] = df["age"].fillna(df["age"].median())


#extracting to a plain numpy array
    X_raw = df[feature_cols].to_numpy(dtype=float)

    # Standardize every attribute to mean 0, variance 1 
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    X = (X_raw - mean) / std


# if sex is female then label A if B then label B
    groups = np.where(df["sex"].to_numpy() == "female", "A", "B")

    return X, groups, feature_cols


if __name__ == "__main__":
    X, groups, feature_cols = make_titanic_fair_kmeans_data()
    print("Features used:", feature_cols)
    print("X shape:", X.shape)
    print("Group A (female):", (groups == "A").sum())
    print("Group B (male):  ", (groups == "B").sum())
    print("Feature means (should be ~0):", X.mean(axis=0).round(3))
    print("Feature stds  (should be ~1):", X.std(axis=0).round(3))
