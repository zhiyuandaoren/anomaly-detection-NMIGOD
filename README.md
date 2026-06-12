# Anomaly Detection Algorithms — CLI Usage Guide

---

## Project Structure

```
anomaly-detection-NMIGOD/
├── ADFNR/                  # Algorithm 1: Fuzzy Neighborhood Rough Set
├── GCN/                    # Algorithm 2: Graph Convolutional Network
├── GCOD/                   # Algorithm 3: Formal Concept Analysis + Granular Computing
├── IE/                     # Algorithm 4: Rough Set Information Entropy
├── KNN/                    # Algorithm 5: K-Nearest Neighbors Distance-Based
├── NIEOD/                  # Algorithm 6: Neighborhood Information Entropy (Numba)
├── NMIGOD/                 # Algorithm 7: Neighborhood Mutual Information + GCN
├── DASOD/                  # Algorithm 8: Dual-View Collaborative FCA
├── datasets/               # 25 benchmark datasets (CSV)
├── images/                 # Output visualizations
│   ├── per_algo/           #   One ROC curve per algorithm (all datasets overlaid)
│   ├── per_dataset/        #   Precision / Recall / F1 / ROC per dataset (all algorithms overlaid)
│   └── summary/            #   Global F1 & ROC summary charts
├── tools/                  # Utility scripts (batch run, metrics collection, plotting, etc.)
└── README.md
```

---

## Basic Usage

- Run without arguments → **Interactive mode** (enter dataset, parameters, etc. step by step)
- Run with arguments → **Command-line mode** (parameters passed directly; suitable for batch execution)

Examples:
```bash
python detector.py                                      # Interactive mode
python detector.py --dataset data.csv --target ...      # Command-line mode
```

---

## Common Parameters (Shared Across All Algorithms)

| Parameter   | Short | Description                                        | Example                  |
|-------------|-------|----------------------------------------------------|--------------------------|
| `--dataset` | `-d`  | Single dataset CSV path                            | `--dataset data.csv`     |
| `--datasets`| `-D`  | Multiple datasets (comma-separated)                | `--datasets a.csv,b.csv` |
| `--target`  | `-t`  | Ground-truth label column name                     | `--target class`         |
| `--anomaly` | `-a`  | Anomaly class value(s) (comma-separated)           | `--anomaly "1,-1"`       |
| `--output`  | `-o`  | Output directory (default: `./output`)             | `--output ./results`     |

> **Note:** Some algorithms accept `--dataset` (single dataset), others accept `--datasets` (multiple datasets, comma-separated).

---

## Algorithm 1: ADFNR — Fuzzy Neighborhood Rough Set Anomaly Detection

- **File**: `ADFNR/detector.py`
- **Mode**: Single dataset (`--dataset`)
- **Additional Parameters**:

| Parameter    | Type  | Description                         | Default |
|--------------|-------|-------------------------------------|---------|
| `--epsilon`  | float | Fuzzy neighborhood radius           | 0.5     |

**Usage Example**:
```bash
python ADFNR/detector.py \
    --dataset datasets/iris.csv \
    --target class \
    --anomaly "Iris-versicolor" \
    --output ./output_iris \
    --epsilon 0.5
```

---

## Algorithm 2: GCN — Graph Convolutional Network Semi-Supervised Classification

- **File**: `GCN/detector.py`
- **Mode**: Multiple datasets (`--datasets`)
- **Additional Parameters**:

| Parameter       | Type  | Description                                    | Default |
|-----------------|-------|------------------------------------------------|---------|
| `--k-neighbors` | int   | Number of neighbors for KNN graph construction | 15      |
| `--hidden1`     | int   | First GCN hidden layer dimension               | 128     |
| `--hidden2`     | int   | Second GCN embedding dimension                 | 64      |
| `--epochs`      | int   | Number of training epochs                      | 200     |
| `--lr`          | float | Learning rate                                  | 0.01    |

**Usage Example**:
```bash
python GCN/detector.py \
    --datasets datasets/iris.csv,datasets/wine.csv \
    --target class \
    --anomaly "Iris-versicolor" \
    --output ./output \
    --k-neighbors 15 --hidden1 128 --hidden2 64 --epochs 200 --lr 0.01
```

---

## Algorithm 3: GCOD — Formal Concept Analysis + Granular Computing Anomaly Detection

- **File**: `GCOD/detector.py`
- **Mode**: Single dataset (`--dataset`)
- **Additional Parameters**:

| Parameter  | Type | Description                        | Default     |
|------------|------|------------------------------------|-------------|
| `--n-jobs` | int  | Number of parallel cores           | auto-detect |

**Usage Example**:
```bash
python GCOD/detector.py \
    -d datasets/adult.csv \
    -t income \
    -a ">50K" \
    -o ./output_adult \
    --n-jobs 4
```

---

## Algorithm 4: IE — Rough Set Information Entropy Anomaly Detection

- **File**: `IE/detector.py`
- **Mode**: Single dataset (`--dataset`)
- **No additional parameters**

**Usage Example**:
```bash
python IE/detector.py \
    -d datasets/german.csv \
    -t Class \
    -a "2" \
    -o ./output_german
```

---

## Algorithm 5: KNN — K-Nearest Neighbors Distance-Based Anomaly Detection

- **File**: `KNN/detector.py`
- **Mode**: Single dataset (`--dataset`)
- **Additional Parameters**:

| Parameter | Type | Description                    | Default |
|-----------|------|--------------------------------|---------|
| `--k`     | int  | Number of nearest neighbors    | 10      |

**Usage Example**:
```bash
python KNN/detector.py \
    -d datasets/glass.csv \
    -t Type_of_glass \
    -a "3,5,6" \
    -o ./output_glass \
    --k 15
```

---

## Algorithm 6: NIEOD — Neighborhood Information Entropy Anomaly Detection (Numba-optimized)

- **File**: `NIEOD/detector.py`
- **Mode**: Single dataset (`--dataset`)
- **Additional Parameters**:

| Parameter   | Type  | Description                              | Default |
|-------------|-------|------------------------------------------|---------|
| `--lambda`  | float | Neighborhood radius adjustment parameter | 1.0     |

**Usage Example**:
```bash
python NIEOD/detector.py \
    -d datasets/wine.csv \
    -t class \
    -a "3" \
    -o ./output_wine \
    --lambda 2.0
```

---

## Algorithm 7: NMIGOD — Neighborhood Mutual Information + GCN Semi-Supervised Anomaly Detection

- **File**: `NMIGOD/detector.py`
- **Mode**: Multiple datasets (`--datasets`)
- **No additional CLI parameters** (algorithm hyperparameters are hardcoded internally)

> Internal defaults: λ = 1.0, graph sparsification threshold d = 0.05, hidden dim = 64, epochs = 200, learning rate = 0.01.

**Usage Example**:
```bash
python NMIGOD/detector.py \
    -D datasets/iris.csv,datasets/bank.csv \
    -t class \
    -a "Iris-setosa" \
    -o ./output
```

---

## Algorithm 8: DASOD — Dual-View Collaborative FCA Anomaly Detection

- **File**: `DASOD/detector.py`
- **Mode**: Multiple datasets (`--datasets`)
- **Additional Parameters**:

| Parameter       | Type  | Description                       | Default |
|-----------------|-------|-----------------------------------|---------|
| `--K`           | int   | Discretization granularity        | 5       |
| `--lambda-ratio`| float | Core concept selection ratio      | 0.05    |

**Usage Example**:
```bash
python DASOD/detector.py \
    -D datasets/adult.csv,datasets/german.csv \
    -t income \
    -a ">50K" \
    -o ./output \
    --K 5 --lambda-ratio 0.1
```

---

## Algorithm Quick Reference

| # | Algorithm | Mode | Extra Parameters |
|---|-----------|------|------------------|
| 1 | ADFNR     | Single (`-d`)  | `--epsilon` (0.5) |
| 2 | GCN       | Multi (`-D`)   | `--k-neighbors` (15), `--hidden1` (128), `--hidden2` (64), `--epochs` (200), `--lr` (0.01) |
| 3 | GCOD      | Single (`-d`)  | `--n-jobs` (auto) |
| 4 | IE        | Single (`-d`)  | — |
| 5 | KNN       | Single (`-d`)  | `--k` (10) |
| 6 | NIEOD     | Single (`-d`)  | `--lambda` (1.0) |
| 7 | NMIGOD    | Multi (`-D`)   | — (hardcoded) |
| 8 | DASOD     | Multi (`-D`)   | `--K` (5), `--lambda-ratio` (0.05) |

---

## Tools (`tools/`)

| Script | Description |
|--------|-------------|
| `run_all_datasets.py` | Batch-run all algorithms on all 25 datasets. Supports `--algo`, `--dataset`, `--cpu`, `--dry-run` flags. |
| `batch_draw.py` | Batch-generate comparison charts (per-algorithm, per-dataset, summary). Supports `--dataset`, `--algo`, `--mode` filters. |
| `collect_metrics.py` | Scan output directories and produce a summary table of Precision, Recall, F1, AUC across all algorithms × datasets. Supports `--base`, `--output`, `--best`. |
| `collect_topk_metrics.py` | Scan output directories and produce a summary table of Top-K anomaly detection metrics across all algorithms × datasets. |
| `collect_params.py` | Scan all detectors and extract default parameter values into a comparison table. |
| `general_framework.py` | General anomaly detection framework template (multi-dataset mode). |
| `image_draw_tool.py` | Draw ROC/Precision/Recall/F1 curves for a single algorithm–dataset pair. |
| `csv_to_xlsx.py` | Convert CSV files to XLSX format (interactive). |
| `xlsx_to_csv.py` | Convert XLSX files to CSV format (interactive). |

---

## Datasets (`datasets/`)

25 benchmark datasets in CSV format:

| Dataset | Target Column | Anomaly Value(s) |
|---------|---------------|------------------|
| adult | income | >50K |
| arrhythmia | C280 | 3,4,5,7,8,9,14,15 |
| bank | y | yes |
| bank-full | y | yes |
| banknote | class | 1 |
| breast-cancer-wisconsin | Class | 4 |
| car | class | good,vgood |
| chess | won | nowin |
| credit | C16 | - |
| diabetes | class | Negative |
| german | Class | 2 |
| glass | Type_of_glass | 3,5,6 |
| horse | cp_data | 1 |
| iris | class | Iris-setosa |
| mushroom | class | m,u,w |
| nursery | class | recommend,very_recom |
| parkinsons | status | 0 |
| raisin | Class | Besni |
| student-mat | G3 | 4,5,7,17,19,20 |
| wine | class | 3 |
| winequality-red | quality | 3,4,8 |
| winequality-white | quality | 3,4,8,9 |
| yeast | Class | ERL |
| zoo | type | 3,5,6 |

---

## Output Files

After processing each dataset, the following files are generated in the output directory:

- `metrics.csv` — Precision, Recall, F1-Score, AUC
- `topk_metrics.csv` — Top-K anomaly detection metrics
- `detection_results.csv` — Anomaly score and detection result for each sample

---

## Batch Execution via `run_all_datasets.py`

```bash
# Run all algorithms on all datasets
python tools/run_all_datasets.py

# Run a specific algorithm only
python tools/run_all_datasets.py --algo ADFNR

# Run on a specific dataset only
python tools/run_all_datasets.py --dataset iris

# Force CPU (disable GPU)
python tools/run_all_datasets.py --cpu

# Dry-run: print the execution plan without actually running
python tools/run_all_datasets.py --dry-run
```

---

## Batch Execution Script Example (Bash)

```bash
#!/bin/bash
# Run ADFNR on all CSV files under datasets/

DATA_DIR="datasets"
OUTPUT_BASE="./batch_results"
TARGET="class"
ANOMALY="1"

for csv in "$DATA_DIR"/*.csv; do
    name=$(basename "$csv" .csv)
    echo "Processing: $name"
    python ADFNR/detector.py \
        -d "$csv" \
        -t "$TARGET" \
        -a "$ANOMALY" \
        -o "$OUTPUT_BASE/adfnr" \
        --epsilon 0.5
done
echo "All done!"
```

---

## Batch Execution Script Example (Python)

```python
import subprocess, os, glob

datasets = glob.glob("datasets/*.csv")
algorithms = {
    "ADFNR":  ["ADFNR/detector.py", "--epsilon", "0.5"],
    "IE":     ["IE/detector.py"],
    "KNN":    ["KNN/detector.py", "--k", "10"],
    "GCN":    ["GCN/detector.py"],
    "NIEOD":  ["NIEOD/detector.py", "--lambda", "1.0"],
}

for algo_name, cmd_base in algorithms.items():
    for csv_path in datasets:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Running {algo_name} on {name}...")
        cmd = [
            "python", cmd_base[0],
            "--dataset" if "--datasets" not in cmd_base else "--datasets",
            csv_path,
            "--target", "class",
            "--anomaly", "1",
            "--output", f"./batch_results/{algo_name}"
        ] + cmd_base[1:]
        subprocess.run(cmd)
```
