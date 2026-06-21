# 异常检测算法 — CLI 使用指南

---

## 项目结构

```
anomaly-detection-NMIGOD/
├── ADFNR/                  # 算法 1: 模糊邻域粗糙集
├── DASOD/                  # 算法 2: 双视角协同形式概念分析
├── GCN/                    # 算法 3: 图卷积网络半监督分类
├── GCN-LOF/                # 算法 4: GCN 嵌入 + LOF 混合异常检测
├── NIEOD/                  # 算法 5: 邻域信息熵 (Numba 加速)
├── NMIGOD/                 # 算法 6: 邻域互信息 + GCN 半监督检测
├── datasets/               # 24 个基准数据集 (CSV)
├── images/                 # 输出可视化
│   ├── per_algo/           #   每个算法一张 ROC 曲线图 (所有数据集叠加)
│   ├── per_dataset/        #   每个数据集一张对比图 (Precision / Recall / F1 / ROC)
│   └── summary/            #   全局 F1 与 ROC 汇总图
├── tools/                  # 工具脚本 (批量运行、指标收集、绘图等)
└── README.md
```

---

## 基本用法

- 不带参数运行 → **交互模式** (逐步输入数据集、参数等)
- 带参数运行 → **命令行模式** (参数直接传入，适合批量执行)

示例：
```bash
python detector.py                                      # 交互模式
python detector.py --dataset data.csv --target ...      # 命令行模式
```

---

## 通用参数 (所有算法共享)

| 参数          | 简写  | 描述                                      | 示例                     |
|---------------|-------|-------------------------------------------|--------------------------|
| `--dataset`   | `-d`  | 单个数据集 CSV 路径                        | `--dataset data.csv`     |
| `--datasets`  | `-D`  | 多个数据集 (逗号分隔)                      | `--datasets a.csv,b.csv` |
| `--target`    | `-t`  | 真实标签列名                               | `--target class`         |
| `--anomaly`   | `-a`  | 异常类别值 (逗号分隔)                      | `--anomaly "1,-1"`       |
| `--output`    | `-o`  | 输出目录 (默认: `./output`)               | `--output ./results`     |

> **注意：** 部分算法使用 `--dataset` (单数据集)，其他使用 `--datasets` (多数据集，逗号分隔)。

---

## 算法 1: ADFNR — 模糊邻域粗糙集异常检测

- **文件**: `ADFNR/detector.py`
- **模式**: 单数据集 (`--dataset`)

**额外参数**：

| 参数         | 类型  | 描述               | 默认值 |
|--------------|-------|--------------------|--------|
| `--epsilon`  | float | 模糊邻域半径       | 0.5    |

**使用示例**：
```bash
python ADFNR/detector.py \
    --dataset datasets/iris.csv \
    --target class \
    --anomaly "Iris-versicolor" \
    --output ./output_iris \
    --epsilon 0.5
```

---

## 算法 2: DASOD — 双视角协同形式概念分析异常检测

- **文件**: `DASOD/detector.py`
- **模式**: 多数据集 (`--datasets`)

**额外参数**：

| 参数              | 类型  | 描述                  | 默认值 |
|-------------------|-------|-----------------------|--------|
| `--K`             | int   | 离散化粒度            | 5      |
| `--lambda-ratio`  | float | 核心概念选择比例      | 0.05   |

**使用示例**：
```bash
python DASOD/detector.py \
    -D datasets/adult.csv,datasets/german.csv \
    -t income \
    -a ">50K" \
    -o ./output \
    --K 5 --lambda-ratio 0.1
```

---

## 算法 3: GCN — 图卷积网络半监督异常检测

- **文件**: `GCN/detector.py`
- **模式**: 多数据集 (`--datasets`)

**额外参数**：

| 参数            | 类型  | 描述                      | 默认值 |
|-----------------|-------|---------------------------|--------|
| `--k-neighbors` | int   | KNN 图构建邻居数           | 15     |
| `--hidden1`     | int   | GCN 第一层隐藏维度         | 128    |
| `--hidden2`     | int   | GCN 第二层嵌入维度         | 64     |
| `--epochs`      | int   | 训练轮数                   | 200    |
| `--lr`          | float | 学习率                     | 0.01   |

> 内部固定参数：`labeled_ratio=0.2`, `random_state=42`, `dropout=0.5`, `weight_decay=5e-4`。

**使用示例**：
```bash
python GCN/detector.py \
    --datasets datasets/iris.csv,datasets/wine.csv \
    --target class \
    --anomaly "Iris-versicolor" \
    --output ./output \
    --k-neighbors 15 --hidden1 128 --hidden2 64 --epochs 200 --lr 0.01
```

---

## 算法 4: GCN-LOF — GCN 嵌入 + LOF 混合异常检测

- **文件**: `GCN-LOF/detector.py`
- **模式**: 多数据集 (`--datasets`)

**额外参数**：

| 参数                  | 类型        | 描述                      | 默认值   |
|-----------------------|-------------|---------------------------|----------|
| `--k-neighbors`       | int         | KNN 图构建邻居数           | 15       |
| `--hidden1`           | int         | GCN 第一层隐藏维度         | 128      |
| `--hidden2`           | int         | GCN 第二层嵌入维度         | 64       |
| `--epochs`            | int         | 训练轮数                   | 200      |
| `--lr`                | float       | 学习率                     | 0.01     |
| `--lof-neighbors`     | int         | LOF 邻居数                 | 20       |
| `--lof-contamination` | str / float | LOF 预期异常比例           | `'auto'` |

> 内部固定参数：`labeled_ratio=0.2`, `random_state=42`, `dropout=0.5`, `weight_decay=5e-4`。

**使用示例**：
```bash
python GCN-LOF/detector.py \
    -D datasets/iris.csv,datasets/wine.csv \
    -t class \
    -a "Iris-versicolor" \
    -o ./output \
    --k-neighbors 15 --hidden1 128 --hidden2 64 --epochs 200 --lr 0.01 \
    --lof-neighbors 20 --lof-contamination auto
```

---

## 算法 5: NIEOD — 邻域信息熵异常检测 (Numba 加速)

- **文件**: `NIEOD/detector.py`
- **模式**: 单数据集 (`--dataset`)

**额外参数**：

| 参数        | 类型  | 描述                   | 默认值 |
|-------------|-------|------------------------|--------|
| `--lambda`  | float | 邻域半径调节参数       | 1.0    |

**使用示例**：
```bash
python NIEOD/detector.py \
    -d datasets/wine.csv \
    -t class \
    -a "3" \
    -o ./output_wine \
    --lambda 2.0
```

---

## 算法 6: NMIGOD — 邻域互信息 + GCN 半监督异常检测

- **文件**: `NMIGOD/detector.py`
- **模式**: 多数据集 (`--datasets`)

**额外参数**：

| 参数              | 类型  | 描述                      | 默认值 |
|-------------------|-------|---------------------------|--------|
| `--lambda-param`  | float | 邻域半径系数              | 1.0    |
| `--hidden1`       | int   | GCN 第一层隐藏维度         | 128    |
| `--hidden2`       | int   | GCN 第二层嵌入维度         | 64     |
| `--epochs`        | int   | 训练轮数                   | 200    |
| `--lr`            | float | 学习率                     | 0.01   |
| `--mi-threshold`  | float | 互信息稀疏化阈值           | 0.05   |

> 内部固定参数：`labeled_ratio=0.2`, `random_state=42`, `dropout=0.5`, `weight_decay=5e-4`。

**使用示例**：
```bash
python NMIGOD/detector.py \
    -D datasets/iris.csv,datasets/bank.csv \
    -t class \
    -a "Iris-setosa" \
    -o ./output \
    --lambda-param 1.0 --mi-threshold 0.05 \
    --hidden1 128 --hidden2 64 --epochs 200 --lr 0.01
```

---

## 算法快速参考

| # | 算法     | 模式            | 额外参数 |
|---|----------|-----------------|----------|
| 1 | ADFNR    | 单数据集 (`-d`) | `--epsilon` (0.5) |
| 2 | DASOD    | 多数据集 (`-D`) | `--K` (5), `--lambda-ratio` (0.05) |
| 3 | GCN      | 多数据集 (`-D`) | `--k-neighbors` (15), `--hidden1` (128), `--hidden2` (64), `--epochs` (200), `--lr` (0.01) |
| 4 | GCN-LOF  | 多数据集 (`-D`) | 同 GCN + `--lof-neighbors` (20), `--lof-contamination` (auto) |
| 5 | NIEOD    | 单数据集 (`-d`) | `--lambda` (1.0) |
| 6 | NMIGOD   | 多数据集 (`-D`) | `--lambda-param` (1.0), `--hidden1` (128), `--hidden2` (64), `--epochs` (200), `--lr` (0.01), `--mi-threshold` (0.05) |

---

## 工具脚本 (`tools/`)

| 脚本 | 描述 |
|------|------|
| `run_all_datasets.py` | 批量运行所有算法在所有 24 个数据集上的检测任务。支持 `--algo`, `--dataset`, `--cpu`, `--dry-run` 参数。 |
| `batch_draw.py` | 批量生成对比图表（per-algorithm, per-dataset, summary）。支持 `--dataset`, `--algo`, `--mode`, `--type`, `-n` 参数。 |
| `collect_metrics.py` | 扫描输出目录，生成所有算法 × 数据集的 Precision / Recall / F1 / AUC 汇总表。支持 `--base`, `--output`, `--best`, `--split`, `-n` 参数。 |
| `collect_topk_metrics.py` | 扫描输出目录，生成所有算法 × 数据集的 Top-K 异常检测指标汇总表。 |
| `collect_params.py` | 扫描所有检测器，提取默认参数值并生成对比表。支持 `--output`, `-n` 参数。 |
| `general_framework.py` | 通用异常检测框架基类（多数据集模式）。各算法继承此类并重写 `train_model()` 和 `get_anomaly_scores()` 方法。 |
| `image_draw_tool.py` | 交互式工具：为单个算法-数据集对绘制 Precision / Recall / F1 曲线和 ROC 曲线。 |
| `csv_to_xlsx.py` | 交互式 CSV → XLSX 格式转换工具。 |
| `xlsx_to_csv.py` | 交互式 XLSX → CSV 格式转换工具（支持单文件与批量文件夹模式）。 |

---

## 数据集 (`datasets/`)

24 个基准数据集 (CSV 格式)：

| 数据集 | 目标列 | 异常值 |
|--------|--------|--------|
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

## 输出文件

每个数据集处理完成后，输出目录中会生成以下文件：

- `metrics.csv` — Precision, Recall, F1-Score, AUC
- `topk_metrics.csv` — Top-K 异常检测指标
- `detection_results.csv` — 每个样本的异常分数与检测结果

---

## 批量执行：`run_all_datasets.py`

```bash
# 运行所有算法在所有数据集上的检测
python tools/run_all_datasets.py

# 仅运行指定算法
python tools/run_all_datasets.py --algo ADFNR

# 仅运行指定数据集
python tools/run_all_datasets.py --dataset iris

# 强制使用 CPU (禁用 GPU)
python tools/run_all_datasets.py --cpu

# 试运行：仅打印执行计划，不实际运行
python tools/run_all_datasets.py --dry-run
```

---

## 批量执行脚本示例 (Bash)

```bash
#!/bin/bash
# 在 datasets/ 下所有 CSV 文件上运行 ADFNR

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

## 批量执行脚本示例 (Python)

```python
import subprocess
import os
import glob

datasets = glob.glob("datasets/*.csv")
algorithms = {
    "ADFNR":   ["ADFNR/detector.py", "--epsilon", "0.5"],
    "NIEOD":   ["NIEOD/detector.py", "--lambda", "1.0"],
    "GCN":     ["GCN/detector.py"],
    "GCN-LOF": ["GCN-LOF/detector.py"],
    "NMIGOD":  ["NMIGOD/detector.py"],
    "DASOD":   ["DASOD/detector.py", "--K", "5", "--lambda-ratio", "0.05"],
}

for algo_name, cmd_base in algorithms.items():
    for csv_path in datasets:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Running {algo_name} on {name}...")
        # 根据算法模式选择 --dataset 或 --datasets
        if algo_name in ("ADFNR", "NIEOD"):
            mode_flag = "--dataset"
        else:
            mode_flag = "--datasets"
        cmd = [
            "python", cmd_base[0],
            mode_flag, csv_path,
            "--target", "class",
            "--anomaly", "1",
            "--output", f"./batch_results/{algo_name}"
        ] + cmd_base[1:]
        subprocess.run(cmd)
```
