"""
Download covertype and skin datasets from UCI, sample 10k instances.
"""
import os, sys, gzip, urllib.request, numpy as np, pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")

def download_covertype_10k():
    """Download covertype from UCI, sample 10k, save."""
    out_path = os.path.join(DATA_DIR, "covertype.csv")
    if os.path.exists(out_path):
        print(f"covertype.csv already exists: {out_path}")
        return

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print("Downloading covertype dataset (~75MB compressed)...")

    gz_path = os.path.join(DATA_DIR, "covtype.data.gz")
    urllib.request.urlretrieve(url, gz_path)

    # Column names from covtype.info
    cols = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ] + [f"Wilderness_Area_{i}" for i in range(4)] + [
        f"Soil_Type_{i}" for i in range(40)
    ] + ["Cover_Type"]

    print("Reading and parsing...")
    with gzip.open(gz_path, 'rt') as f:
        df = pd.read_csv(f, header=None, names=cols)

    print(f"Full dataset: {df.shape}")

    # Sample 10k with stratification on Cover_Type
    df_10k = df.groupby("Cover_Type", group_keys=False).apply(
        lambda x: x.sample(n=max(1, int(len(x) * 10000 / len(df))), random_state=42)
    ).reset_index(drop=True)

    # If > 10000 due to rounding, trim
    if len(df_10k) > 10000:
        df_10k = df_10k.sample(n=10000, random_state=42).reset_index(drop=True)

    # 标准化标签: Cover_Type (4,5=anomaly) → class (anomaly/normal)
    anomaly_mask = df_10k["Cover_Type"].isin([4, 5])
    df_10k["class"] = np.where(anomaly_mask, "anomaly", "normal")
    df_10k = df_10k.drop(columns=["Cover_Type"])
    df_10k.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_10k)} instances, anomaly={(df_10k['class']=='anomaly').sum()})")

    # Clean up gz
    os.remove(gz_path)


def download_skin_10k():
    """Download skin segmentation from UCI, sample 10k, save."""
    out_path = os.path.join(DATA_DIR, "skin.csv")
    if os.path.exists(out_path):
        print(f"skin.csv already exists: {out_path}")
        return

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt"
    print("Downloading skin segmentation dataset (~8MB)...")

    txt_path = os.path.join(DATA_DIR, "Skin_NonSkin.txt")
    urllib.request.urlretrieve(url, txt_path)

    print("Reading and parsing...")
    df = pd.read_csv(txt_path, header=None, names=["B", "G", "R", "class"], delim_whitespace=True)

    print(f"Full dataset: {df.shape}")
    print(f"Class distribution: {df['class'].value_counts().to_dict()}")

    # Sample 10k stratified
    df_10k = df.groupby("class", group_keys=False).apply(
        lambda x: x.sample(n=max(1, int(len(x) * 10000 / len(df))), random_state=42)
    ).reset_index(drop=True)

    if len(df_10k) > 10000:
        df_10k = df_10k.sample(n=10000, random_state=42).reset_index(drop=True)

    # 标准化标签: class (1=anomaly, 2=normal) → class (anomaly/normal)
    df_10k["class"] = np.where(df_10k["class"] == 1, "anomaly", "normal")
    df_10k.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_10k)} instances, anomaly={(df_10k['class']=='anomaly').sum()})")

    os.remove(txt_path)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_covertype_10k()
    download_skin_10k()
    print("Done!")
