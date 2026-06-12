```markdown
# 异常检测算法命令行调用指南
## Command-Line Interface (CLI) Usage Guide

---

## 基本使用方式

- 无参数运行 → 交互式模式（交互式输入数据集、参数等）
- 带参数运行 → 命令行模式（参数直接传入，可直接批量运行）

示例：
```bash
python detector.py                                      # 交互模式
python detector.py --dataset data.csv --target ...      # 命令行模式
```

---

## 通用参数（所有算法共用）

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--dataset` | `-d` | 单个数据集CSV路径 | `--dataset data.csv` |
| `--datasets` | `-D` | 多个数据集（逗号分隔） | `--datasets a.csv,b.csv` |
| `--target` | `-t` | 真实标签列名 | `--target class` |
| `--anomaly` | `-a` | 异常值（逗号分隔） | `--anomaly "1,-1"` |
| `--output` | `-o` | 输出文件夹（默认`./output`） | `--output ./results` |

---

## 算法 1: ADFNR — 模糊邻域粗糙集异常检测

- **文件**: `ADFNR/detector.py`
- **模式**: 单数据集 (`--dataset`)
- **额外参数**:
  - `--epsilon` (float): 模糊邻域半径（默认: 0.5）

**调用示例**:
```bash
python ADFNR/detector.py \
    --dataset datasets/iris.csv \
    --target class \
    --anomaly "Iris-versicolor" \
    --output ./output_iris \
    --epsilon 0.5
```

Windows PowerShell:
```powershell
python ADFNR/detector.py -d datasets/iris.csv -t class -a "Iris-versicolor" -o ./output_iris --epsilon 0.5
```

---

## 算法 2: GCN — 图卷积网络半监督分类

- **文件**: `GCN/detector.py`
- **模式**: 多数据集 (`--datasets`)
- **额外参数**:
  - `--knn-k` (int): KNN图构建近邻数（默认: 10）
  - `--train-ratio` (float): 训练集比例（默认: 0.5）

**调用示例**:
```bash
python GCN/detector.py \
    --datasets datasets/iris.csv,datasets/wine.csv \
    --target class \
    --anomaly "Iris-versicolor" \
    --output ./output \
    --knn-k 15
```

---

## 算法 3: GCN-LOF — 图卷积网络+局部异常因子

- **文件**: `GCN-LOF/detector.py`
- **模式**: 单数据集 (`--dataset`)
- **额外参数**:
  - `--hidden-dim` (int): 隐藏层维度（默认: 16）
  - `--alpha` (float): 相似度权重α（默认: 0.5）
  - `--beta` (float): LOF影响因子β（默认: 1.0）

**调用示例**:
```bash
python GCN-LOF/detector.py \
    -d datasets/bank.csv \
    -t deposit \
    -a "yes" \
    -o ./output_bank \
    --hidden-dim 32 --alpha 0.5 --beta 1.0
```

---

## 算法 4: GCOD — 形式概念分析+粒度计算异常检测

- **文件**: `GCOD/detector.py`
- **模式**: 单数据集 (`--dataset`)
- **额外参数**:
  - `--n-jobs` (int): 并行核心数（默认: 自动检测）

**调用示例**:
```bash
python GCOD/detector.py \
    -d datasets/adult.csv \
    -t income \
    -a ">50K" \
    -o ./output_adult \
    --n-jobs 4
```

---

## 算法 5: IE — 粗糙集信息熵异常检测

- **文件**: `IE/detector.py`
- **模式**: 单数据集 (`--dataset`)
- **无额外参数**

**调用示例**:
```bash
python IE/detector.py \
    -d datasets/german.csv \
    -t credit_risk \
    -a "bad" \
    -o ./output_german
```

---

## 算法 6: KNN — K近邻距离异常检测

- **文件**: `KNN/detector.py`
- **模式**: 单数据集 (`--dataset`)
- **额外参数**:
  - `--k` (int): K近邻数（默认: 10）

**调用示例**:
```bash
python KNN/detector.py \
    -d datasets/glass.csv \
    -t type \
    -a "1,2,3" \
    -o ./output_glass \
    --k 15
```

---

## 算法 7: NIEOD — 邻域信息熵异常检测 (Numba优化)

- **文件**: `NIEOD/detector.py`
- **模式**: 单数据集 (`--dataset`)
- **额外参数**:
  - `--lambda` (float): 邻域半径调节参数λ（默认: 1.0）

**调用示例**:
```bash
python NIEOD/detector.py \
    -d datasets/wine.csv \
    -t class \
    -a "1" \
    -o ./output_wine \
    --lambda 2.0
```

---

## 算法 8: NMIGOD — 邻域互信息+GCN半监督异常检测

- **文件**: `NMIGOD/detector.py`
- **模式**: 多数据集 (`--datasets`)
- **额外参数**:
  - `--lambda` (float): 邻域半径参数λ（默认: 1.0）
  - `--threshold-d` (float): 图稀疏化阈值d（默认: 0.05）
  - `--epochs` (int): GCN训练轮数（默认: 200）
  - `--lr` (float): 学习率η（默认: 0.01）
  - `--label-rate` (float): 半监督标注比例（默认: 0.2）

**调用示例**:
```bash
python NMIGOD/detector.py \
    -D datasets/iris.csv,datasets/bank.csv \
    -t class \
    -a "Iris-versicolor" \
    -o ./output \
    --epochs 300 --lr 0.005
```

---

## 算法 9: DASOD — 双视角协同FCA异常检测

- **文件**: `DASOD/detector.py`
- **模式**: 多数据集 (`--datasets`)
- **额外参数**:
  - `--K` (int): 离散化粒度（默认: 5）
  - `--lambda-ratio` (float): 核心概念选择比例λ（默认: 0.05）

**调用示例**:
```bash
python DASOD/detector.py \
    -D datasets/adult.csv,datasets/german.csv \
    -t income \
    -a ">50K" \
    -o ./output \
    --K 5 --lambda-ratio 0.1
```

---

## 框架模板: General Framework

- **文件**: `tools/general_framework.py`
- **模式**: 多数据集 (`--datasets`)

**调用示例**:
```bash
python tools/general_framework.py \
    -D datasets/iris.csv \
    -t class \
    -a "Iris-setosa" \
    -o ./output
```

---

## 批量运行脚本示例 (Bash)

```bash
#!/bin/bash
# 对 datasets/ 下所有 CSV 文件运行 ADFNR

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

## 批量运行脚本示例 (Python)

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

---

## 输出文件说明

每个数据集处理完成后，会在输出文件夹中生成：

- `metrics.csv` — Precision, Recall, F1-Score, AUC
- `topk_metrics.csv` — Top-K 异常检测指标
- `detection_results.csv` — 每个样本的异常分数和检测结果

```