"""
One-time data-prep script (run with Python once) to build LFW.mat for the
MATLAB Fair-K-Means-Clustering repo's loadData.m, which expects:
    ../Data/LFW/LFW.mat  containing variables 'data' (n x 1764) and
    'sensitive' (n x 1, values 1=Female / 2=Male)

Requires: pip install numpy pandas opencv-python scipy

NOTE: the original host vis-www.cs.umass.edu appears to be down (this is a
known, ongoing issue -- scikit-learn and torchvision both had to switch to
mirrors for the same reason). This script therefore uses:
  images:     the Figshare mirror scikit-learn itself uses for the *same*
              original lfw.tgz file (identical bytes, same real photographs)
  attributes: tries the original UMass URL first, then falls back to the
              Columbia University CAVE lab mirror of the same file
"""

import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import cv2
from scipy.io import savemat

IMAGES_URL = "https://ndownloader.figshare.com/files/5976018"  # = official lfw.tgz, mirrored
ATTRS_URLS = [
    "http://vis-www.cs.umass.edu/lfw/lfw_attributes.txt",
    "http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt",
]

IMAGES_PATH = "lfw.tgz"
ATTRS_PATH = "lfw_attributes.txt"

# Paper says images are 49x36 (vector dim 1764 = 49*36)
OUT_H, OUT_W = 49, 36

HEADERS = {"User-Agent": "Mozilla/5.0 (research data download script)"}

def download(url, path):
    if os.path.exists(path):
        print(f"Found {path} already on disk, skipping download.")
        return True
    try:
        print(f"Downloading {url} ...")
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as out:
            out.write(resp.read())
        return True
    except Exception as e:
        print(f"  failed: {e}")
        return False

def download_any(urls, path):
    if os.path.exists(path):
        print(f"Found {path} already on disk, skipping download.")
        return True
    for url in urls:
        if download(url, path):
            return True
    return False

def main():
    if not download(IMAGES_URL, IMAGES_PATH):
        raise SystemExit(
            "Could not download the LFW image archive. Check your internet "
            "connection / firewall, or try downloading it manually from "
            "https://ndownloader.figshare.com/files/5976018 and place it "
            "next to this script as 'lfw.tgz'."
        )
    if not download_any(ATTRS_URLS, ATTRS_PATH):
        raise SystemExit(
            "Could not download the gender-attributes file from either "
            "mirror. Try downloading lfw_attributes.txt manually (search "
            "'lfw_attributes.txt download') and place it next to this "
            "script."
        )

    # --- parse attributes file ---
    df_attrs = pd.read_csv(ATTRS_PATH, sep="\t", skiprows=1)
    df_attrs = pd.DataFrame(
        df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:]
    )
    df_attrs["imagenum"] = df_attrs["imagenum"].astype(int)

    male_score = df_attrs["Male"].astype(float)
    print("Male-score sanity check -> fraction predicted male:",
          (male_score > 0).mean(), "(LFW is known to be ~77% male)")

    lookup = {}
    for _, row in df_attrs.iterrows():
        key = (row["person"], int(row["imagenum"]))
        lookup[key] = 2 if float(row["Male"]) > 0 else 1  # 2=Male, 1=Female

    # --- walk the image tarball, crop/resize, pull matching gender label ---
    all_vecs = []
    all_labels = []
    with tarfile.open(IMAGES_PATH) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg")]
        print(f"{len(members)} images found in tarball")
        for m in members:
            fname = os.path.basename(m.name)
            stem = fname[:-4]
            parts = stem.replace("_", " ").split()
            person = " ".join(parts[:-1])
            imagenum = int(parts[-1])
            key = (person, imagenum)
            if key not in lookup:
                continue

            raw = tf.extractfile(m).read()
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            crop = img[int(h*0.1):int(h*0.9), int(w*0.15):int(w*0.85)]
            resized = cv2.resize(crop, (OUT_W, OUT_H))
            all_vecs.append(resized.flatten().astype(np.float64))
            all_labels.append(lookup[key])

    data = np.stack(all_vecs)
    sensitive = np.array(all_labels).reshape(-1, 1)

    print("Final data shape:", data.shape, " sensitive shape:", sensitive.shape)
    print("Label counts -> Female(1):", (sensitive == 1).sum(),
          " Male(2):", (sensitive == 2).sum())

    os.makedirs("LFW_out", exist_ok=True)
    savemat("LFW_out/LFW.mat", {"data": data, "sensitive": sensitive})
    print("Saved LFW_out/LFW.mat -- copy this into Fair-K-Means-Clustering/Data/LFW/")

if __name__ == "__main__":
    main()