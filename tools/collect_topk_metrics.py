"""
收集所有算法的 topk_metrics.csv，生成汇总表格。

输出格式：
  数据集 | k(个数) | ADFNR_Precision | ADFNR_Recall | DASOD_Precision | DASOD_Recall | ...

用法：
  cd <project_root>
  python tools/collect_topk_metrics.py
"""

import csv
import os
from pathlib import Path

# 项目根目录（脚本所在目录的上级）
BASE_DIR = Path(__file__).resolve().parent.parent

# 算法目录名列表（按字母排序，保证列的顺序稳定）
ALGORITHMS = ["ADFNR", "DASOD", "GCN", "GCOD", "IE", "KNN", "NIEOD", "NMIGOD"]


def find_all_datasets(base_dir: Path, algorithms: list[str]) -> list[str]:
    """扫描第一个算法目录，获取所有数据集名称（假设各算法数据集一致）。"""
    first_alg = algorithms[0]
    pattern = f"output_*"
    output_dirs = sorted(base_dir.glob(f"{first_alg}/{pattern}"))
    datasets = [d.name.replace("output_", "") for d in output_dirs if d.is_dir()]
    return datasets


def load_csv(filepath: Path) -> dict[str, dict[str, str]]:
    """
    读取单个 topk_metrics.csv，返回 { k个数: {"Precision": ..., "Recall": ...} }。
    k 取 Top_K 列（异常样本个数）。
    """
    rows = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row["Top_K"].strip()
            precision = row["Precision"].strip()
            recall = row["Recall"].strip()
            rows[k] = {"Precision": precision, "Recall": recall}
    return rows


def collect_all(
    base_dir: Path, algorithms: list[str], datasets: list[str]
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """
    返回嵌套结构：
      { dataset: { k%: { algorithm: {"Precision": ..., "Recall": ...} } } }

    对不存在的文件，对应算法指标的值为 "-"。
    """
    all_data: dict[str, dict[str, dict[str, dict[str, str]]]] = {}

    for dataset in datasets:
        all_data[dataset] = {}

        for alg in algorithms:
            filepath = base_dir / alg / f"output_{dataset}" / "topk_metrics.csv"
            if filepath.is_file():
                alg_rows = load_csv(filepath)
                for k, metrics in alg_rows.items():
                    all_data[dataset].setdefault(k, {})[alg] = metrics
            else:
                # 文件缺失时，为所有已有的 k 填 "-"
                for k in all_data[dataset]:
                    all_data[dataset][k][alg] = {"Precision": "-", "Recall": "-"}

    return all_data


NUM_SELECT = 9  # 每个数据集选取的 k 值数量


def _select_indices(n: int, k: int) -> list[int]:
    """从 n 个元素中均匀选取 k 个索引。n <= k 时返回全部索引。"""
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def write_csv(
    output_path: Path,
    algorithms: list[str],
    datasets: list[str],
    all_data: dict,
) -> None:
    """写出汇总 CSV。每个数据集选取 NUM_SELECT 个 k 值，末尾追加平均值行。"""
    # 构建表头
    header = ["Dataset", "Top_K"]
    for alg in algorithms:
        header.append(f"{alg}_Precision")
        header.append(f"{alg}_Recall")

    total_rows = 0

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for dataset in datasets:
            ds_data = all_data.get(dataset, {})
            sorted_k = sorted(ds_data.keys(), key=int)

            # 均匀选取 NUM_SELECT 个 k 值
            selected_indices = _select_indices(len(sorted_k), NUM_SELECT)
            selected_k = [sorted_k[i] for i in selected_indices]

            # 累加器，用于计算平均值
            avg_accum: dict[str, dict[str, list[float]]] = {
                alg: {"Precision": [], "Recall": []} for alg in algorithms
            }

            for k in selected_k:
                row = [dataset, k]
                alg_metrics = ds_data.get(k, {})
                for alg in algorithms:
                    metrics = alg_metrics.get(alg, {"Precision": "-", "Recall": "-"})
                    p = metrics["Precision"]
                    r = metrics["Recall"]
                    row.append(p)
                    row.append(r)
                    # 收集有效数值用于算平均
                    if p != "-":
                        avg_accum[alg]["Precision"].append(float(p))
                    if r != "-":
                        avg_accum[alg]["Recall"].append(float(r))
                writer.writerow(row)
                total_rows += 1

            # 写入平均值行
            avg_row = [f"{dataset}_Avg", "-"]
            for alg in algorithms:
                p_vals = avg_accum[alg]["Precision"]
                r_vals = avg_accum[alg]["Recall"]
                avg_p = f"{sum(p_vals) / len(p_vals):.4f}" if p_vals else "-"
                avg_r = f"{sum(r_vals) / len(r_vals):.4f}" if r_vals else "-"
                avg_row.append(avg_p)
                avg_row.append(avg_r)
            writer.writerow(avg_row)
            total_rows += 1

    print(f"汇总完成，共 {len(datasets)} 个数据集，每数据集 {NUM_SELECT} 条 + 1 平均值行，总计 {total_rows} 行")
    print(f"输出文件：{output_path}")


def main():
    datasets = find_all_datasets(BASE_DIR, ALGORITHMS)
    print(f"发现 {len(datasets)} 个数据集：{datasets}")
    print(f"算法：{ALGORITHMS}")

    all_data = collect_all(BASE_DIR, ALGORITHMS, datasets)

    output_path = BASE_DIR / "tools" / "all_topk_metrics.csv"
    write_csv(output_path, ALGORITHMS, datasets, all_data)


if __name__ == "__main__":
    main()
