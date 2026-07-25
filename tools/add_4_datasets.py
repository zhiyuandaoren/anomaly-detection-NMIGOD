"""
Download and prepare 4 new mixed-attribute UCI datasets for anomaly detection.
Adds: abalone, heart, cmc, hepatitis
"""
import os, sys, urllib.request, pandas as pd, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

NEW_DATASETS = []

# ============================================================
# 1. Abalone — predict age from physical measurements
#    Mixed: 1 cat (Sex) + 7 num, 4177 instances
# ============================================================
def prepare_abalone():
    name = "abalone"
    out_path = DATA_DIR / f"{name}.csv"
    if out_path.exists():
        print(f"[SKIP] {name}.csv already exists")
        return True

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    cols = ["Sex", "Length", "Diameter", "Height", "Whole_weight",
            "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
    df = pd.read_csv(url, header=None, names=cols)
    print(f"[{name}] Downloaded: {df.shape}")

    # Anomaly: very young (Rings <= 4) or very old (Rings >= 20)
    df["class"] = df["Rings"].apply(
        lambda x: "anomaly" if x <= 4 or x >= 20 else "normal")
    df = df.drop(columns=["Rings"])
    df.to_csv(out_path, index=False)
    print(f"[{name}] Saved: {len(df)} rows, anomaly={(df['class']=='anomaly').sum()}")
    NEW_DATASETS.append({
        "name": name, "outlier_col": "class", "outlier_val": "anomaly",
        "num_cols": 7, "cat_cols": 1, "samples": len(df),
        "anomalies": int((df['anomaly_label']=='anomaly').sum()),
        "ratio": f"{(df['anomaly_label']=='anomaly').mean()*100:.2f}%", "dtype": "Mixed"
    })
    return True


# ============================================================
# 2. Heart Disease (Cleveland) — medical diagnosis
#    Mixed: 7 num + 6 cat, 303 instances
# ============================================================
def prepare_heart():
    name = "heart"
    out_path = DATA_DIR / f"{name}.csv"
    if out_path.exists():
        print(f"[SKIP] {name}.csv already exists")
        return True

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    df = pd.read_csv(url, header=None, names=cols, na_values="?")

    # Clean missing values
    for c in ["ca", "thal"]:
        df[c] = df[c].fillna(df[c].mode()[0] if len(df[c].mode()) > 0 else 0)
    df = df.dropna()

    # Anomaly: heart disease present (num > 0)
    df["class"] = df["num"].apply(
        lambda x: "anomaly" if x > 0 else "normal")
    df = df.drop(columns=["num"])
    df.to_csv(out_path, index=False)
    print(f"[{name}] Saved: {len(df)} rows, anomaly={(df['class']=='anomaly').sum()}")
    NEW_DATASETS.append({
        "name": name, "outlier_col": "class", "outlier_val": "anomaly",
        "num_cols": 7, "cat_cols": 6, "samples": len(df),
        "anomalies": int((df['anomaly_label']=='anomaly').sum()),
        "ratio": f"{(df['anomaly_label']=='anomaly').mean()*100:.2f}%", "dtype": "Mixed"
    })
    return True


# ============================================================
# 3. Contraceptive Method Choice (CMC) — demographic survey
#    Mixed: 2 num + 7 cat, 1473 instances
# ============================================================
def prepare_cmc():
    name = "cmc"
    out_path = DATA_DIR / f"{name}.csv"
    if out_path.exists():
        print(f"[SKIP] {name}.csv already exists")
        return True

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data"
    cols = ["wife_age", "wife_edu", "husband_edu", "num_children",
            "wife_religion", "wife_working", "husband_occupation",
            "living_index", "media_exposure", "method"]
    df = pd.read_csv(url, header=None, names=cols)

    # Anomaly: long-term contraceptive methods (class 1 = no-use, class 2 = long-term)
    # Normal: short-term methods (class 3)
    df["class"] = df["method"].apply(
        lambda x: "anomaly" if x in [1] else "normal")
    df = df.drop(columns=["method"])
    df.to_csv(out_path, index=False)
    print(f"[{name}] Saved: {len(df)} rows, anomaly={(df['class']=='anomaly').sum()}")
    NEW_DATASETS.append({
        "name": name, "outlier_col": "class", "outlier_val": "anomaly",
        "num_cols": 2, "cat_cols": 7, "samples": len(df),
        "anomalies": int((df['anomaly_label']=='anomaly').sum()),
        "ratio": f"{(df['anomaly_label']=='anomaly').mean()*100:.2f}%", "dtype": "Mixed"
    })
    return True


# ============================================================
# 4. Hepatitis — survival prediction
#    Mixed: 6 num + 13 cat, 155 instances
# ============================================================
def prepare_hepatitis():
    name = "hepatitis"
    out_path = DATA_DIR / f"{name}.csv"
    if out_path.exists():
        print(f"[SKIP] {name}.csv already exists")
        return True

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
    cols = ["Class", "AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE",
            "MALAISE", "ANOREXIA", "LIVER_BIG", "LIVER_FIRM", "SPLEEN_PALPABLE",
            "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK_PHOSPHATE",
            "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"]
    df = pd.read_csv(url, header=None, names=cols, na_values="?")

    # Fill missing numerical values with mean, categorical with mode
    num_cols_hep = ["AGE", "BILIRUBIN", "ALK_PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME"]
    cat_cols_hep = [c for c in cols if c not in num_cols_hep and c != "Class"]
    for c in num_cols_hep:
        df[c] = df[c].fillna(df[c].mean())
    for c in cat_cols_hep:
        df[c] = df[c].fillna(df[c].mode()[0] if len(df[c].mode()) > 0 else 1)

    # Anomaly: died (Class=1), normal: lived (Class=2)
    df["class"] = df["Class"].apply(
        lambda x: "anomaly" if x == 1 else "normal")
    df = df.drop(columns=["Class"])
    df.to_csv(out_path, index=False)
    print(f"[{name}] Saved: {len(df)} rows, anomaly={(df['class']=='anomaly').sum()}")
    NEW_DATASETS.append({
        "name": name, "outlier_col": "class", "outlier_val": "anomaly",
        "num_cols": 6, "cat_cols": 13, "samples": len(df),
        "anomalies": int((df['anomaly_label']=='anomaly').sum()),
        "ratio": f"{(df['anomaly_label']=='anomaly').mean()*100:.2f}%", "dtype": "Mixed"
    })
    return True


# ============================================================
# Update datasets_config.csv
# ============================================================
def update_config():
    config_path = BASE / "datasets" / "datasets_config.csv"
    # Read existing config
    with open(config_path, "r", encoding="gbk") as f:
        content = f.read()

    lines = content.strip().split("\n")
    # Find last data row
    last_idx = 0
    for i, line in enumerate(lines):
        parts = line.split(",")
        if len(parts) > 1 and parts[0].strip().isdigit():
            last_idx = i

    next_no = int(lines[last_idx].split(",")[0].strip()) + 1
    print(f"Next dataset number: {next_no}")

    new_rows = []
    for ds in NEW_DATASETS:
        row = f'{next_no},{ds["name"]},{ds["samples"]},{ds["num_cols"]+ds["cat_cols"]},{ds["anomalies"]},{ds["ratio"]},{ds["dtype"]},{ds["outlier_col"]},{ds["outlier_val"]}'
        new_rows.append(row)
        next_no += 1

    # Insert before last empty/notes lines
    insert_pos = last_idx + 1
    new_lines = lines[:insert_pos] + new_rows + [""]
    new_content = "\n".join(new_lines)

    with open(config_path, "w", encoding="gbk") as f:
        f.write(new_content)
    print(f"Updated datasets_config.csv with {len(new_rows)} new datasets")


# ============================================================
# Update run_all_datasets.py DATASETS list
# ============================================================
def update_run_script():
    script_path = BASE / "tools" / "run_all_datasets.py"
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if already added
    for ds in NEW_DATASETS:
        if f'"name": "{ds["name"]}"' in content:
            print(f"[SKIP] {ds['name']} already in run_all_datasets.py")
            continue
        # Insert before the last dataset entry (yeast)
        insert_marker = '{"name": "zoo"'
        new_entry = f'    {{"name": "{ds["name"]}", "outlier_col": "class", "outlier_val": "anomaly"}},\n'
        content = content.replace(insert_marker, new_entry + "    " + insert_marker)
        print(f"Added {ds['name']} to run_all_datasets.py")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading 4 new mixed-attribute UCI datasets...")
    print("=" * 60)

    ok = all([
        prepare_abalone(),
        prepare_heart(),
        prepare_cmc(),
        prepare_hepatitis(),
    ])

    if ok and NEW_DATASETS:
        print("\n" + "=" * 60)
        print("New datasets summary:")
        for ds in NEW_DATASETS:
            print(f"  {ds['name']:<15s} {ds['samples']:>5d} rows  "
                  f"anomaly={ds['anomalies']:>4d} ({ds['ratio']})  "
                  f"num={ds['num_cols']} cat={ds['cat_cols']}")
        print("=" * 60)

        update_config()
        update_run_script()
        print("\nAll done! Ready to run algorithms.")
    else:
        print("\nAll datasets already exist (nothing to download).")
