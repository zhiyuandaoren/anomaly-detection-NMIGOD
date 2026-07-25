#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 30 个数据集的标签列统一为 "class"，标签值统一为 "normal"/"anomaly"。
同时修复 datasets_config.csv 和 run_all_datasets.py 中的已知问题。

用法:
  python tools/standardize_labels.py            # 执行转换
  python tools/standardize_labels.py --verify   # 仅验证, 不修改
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets"

# ============================================================
# 文件名映射: 配置名 → 实际文件名 (不含 .csv)
# ============================================================
FILE_NAME_MAP = {
    "breast-cancer": "breast-cancer-wisconsin",
    "wine-red":     "winequality-red",
    "wine-white":   "winequality-white",
}


def resolve_filename(name):
    """将配置名转为实际文件名 (不含扩展名)."""
    return FILE_NAME_MAP.get(name, name)


# ============================================================
# 每个数据集的转换规则
#   label_col: 当前标签列名
#   anomaly_values: 异常值列表 (字符串形式, 用于精确匹配)
#   不在 anomaly_values 中的值 → "normal"
# ============================================================
DATASET_RULES = [
    # (config_name, label_col, anomaly_values)
    ("adult",          "income",        [">50K"]),
    ("arrhythmia",     "C280",          ["3", "4", "5", "7", "8", "9", "14", "15"]),
    ("bank",           "y",             ["yes"]),
    ("bank-full",      "y",             ["yes"]),
    ("banknote",       "class",         ["1"]),
    ("breast-cancer",  "Class",         ["4"]),
    ("car",            "class",         ["good", "vgood"]),
    ("chess",          "won",           ["nowin"]),
    ("covertype",      "Cover_Type",    ["4", "5"]),
    ("credit",         "C16",           ["-"]),
    ("diabetes",       "class",         ["Negative"]),
    ("german",         "Class",         ["2"]),
    ("glass",          "Type_of_glass", ["3", "5", "6"]),
    ("horse",          "cp_data",       ["1"]),
    ("iris",           "class",         ["Iris-setosa"]),
    ("mushroom",       "class",         ["m", "u", "w"]),
    ("nursery",        "class",         ["recommend", "very_recom"]),
    ("parkinsons",     "status",        ["0"]),
    ("raisin",         "Class",         ["Besni"]),
    ("skin",           "class",         ["1"]),
    ("student-mat",    "G3",            ["4", "5", "7", "17", "19", "20"]),
    ("wine",           "class",         ["3"]),
    ("wine-red",       "quality",       ["3", "4", "8"]),
    ("wine-white",     "quality",       ["3", "4", "8", "9"]),
    ("yeast",          "Class",         ["ERL"]),
    ("zoo",            "type",          ["3", "5", "6"]),
    # 以下 4 个数据集已经使用 anomaly_label + anomaly/normal 格式
    ("abalone",        "anomaly_label", ["anomaly"]),
    ("heart",          "anomaly_label", ["anomaly"]),
    ("cmc",            "anomaly_label", ["anomaly"]),
    ("hepatitis",      "anomaly_label", ["anomaly"]),
]


def transform_dataset(config_name, label_col, anomaly_values, dry_run=False):
    """
    对单个数据集执行标签标准化:
      1. 将标签列重命名为 "class" (如需)
      2. 将异常值映射为 "anomaly", 其余值映射为 "normal"
    """
    actual_name = resolve_filename(config_name)
    csv_path = DATA_DIR / f"{actual_name}.csv"

    if not csv_path.exists():
        return False, f"FILE NOT FOUND: {csv_path}"

    df = pd.read_csv(csv_path)
    n_total = len(df)

    # 验证标签列存在
    if label_col not in df.columns:
        return False, f"列 '{label_col}' 不在 {list(df.columns)[:10]}... 中"

    # --- 映射值 ---
    col_values = df[label_col].copy()

    # 统一转为字符串进行匹配 (strip 去除首尾空格)
    str_values = col_values.astype(str).str.strip()

    # 构建布尔掩码: True = 异常
    anomaly_set = set(anomaly_values)
    is_anomaly = str_values.isin(anomaly_set)

    n_anomaly_before = is_anomaly.sum()
    n_normal_before = n_total - n_anomaly_before

    # 赋值新标签
    new_labels = np.where(is_anomaly, "anomaly", "normal")

    # --- 处理列名 ---
    if label_col == "class":
        # 列名已正确, 直接替换值
        df["class"] = new_labels
    elif label_col == "Class":
        # 列名仅大小写不同, 改为小写
        df.drop(columns=["Class"], inplace=True)
        df["class"] = new_labels
    else:
        # 列名完全不同, 需要重命名
        df.drop(columns=[label_col], inplace=True)
        df["class"] = new_labels

    if not dry_run:
        df.to_csv(csv_path, index=False)

    return True, (n_total, n_anomaly_before, n_normal_before)


def run_transformation(dry_run=False):
    """对所有 30 个数据集执行标准化."""
    results = []
    errors = []

    print(f"{'[DRY-RUN] ' if dry_run else ''}开始标准化 {len(DATASET_RULES)} 个数据集...")
    print(f"{'='*70}")

    for config_name, label_col, anomaly_values in DATASET_RULES:
        success, info = transform_dataset(config_name, label_col,
                                          anomaly_values, dry_run=dry_run)

        if success:
            n_total, n_anom, n_norm = info
            pct = n_anom / n_total * 100
            status = "OK"
            results.append((config_name, n_total, n_anom, n_norm, pct))
            print(f"  [{status}] {config_name:<18s}  "
                  f"{n_total:>6d} rows  anomaly={n_anom:>5d} ({pct:>5.1f}%)  "
                  f"normal={n_norm:>5d}")
        else:
            errors.append((config_name, info))
            print(f"  [FAIL] {config_name:<18s}  {info}")

    print(f"{'='*70}")
    print(f"成功: {len(results)}/{len(DATASET_RULES)}, "
          f"失败: {len(errors)}")

    if errors:
        print(f"\n失败详情:")
        for name, err in errors:
            print(f"  - {name}: {err}")

    return len(errors) == 0


def verify_datasets():
    """验证所有数据集是否已标准化."""
    print("=== 验证数据集标签标准化状态 ===\n")
    all_ok = True

    for config_name, label_col, anomaly_values in DATASET_RULES:
        actual_name = resolve_filename(config_name)
        csv_path = DATA_DIR / f"{actual_name}.csv"

        if not csv_path.exists():
            print(f"  [MISS] {csv_path}")
            all_ok = False
            continue

        df = pd.read_csv(csv_path)

        # 检查 class 列
        if "class" not in df.columns:
            print(f"  [FAIL] {config_name}: 缺少 'class' 列, "
                  f"当前列: {list(df.columns)[:5]}...")
            all_ok = False
            continue

        # 检查值
        unique_vals = set(df["class"].unique())
        expected = {"normal", "anomaly"}
        if unique_vals != expected:
            print(f"  [FAIL] {config_name}: class 值 = {unique_vals}, "
                  f"期望 = {expected}")
            all_ok = False
            continue

        n_anomaly = (df["class"] == "anomaly").sum()
        n_total = len(df)
        pct = n_anomaly / n_total * 100
        print(f"  [OK]  {config_name:<18s}  {n_total:>6d} rows  "
              f"anomaly={n_anomaly:>5d} ({pct:>5.1f}%)")

    print(f"\n{'='*50}")
    if all_ok:
        print("所有数据集验证通过!")
    else:
        print("存在未标准化的数据集, 请重新运行转换脚本.")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="标准化数据集标签列 → class, 值 → normal/anomaly")
    parser.add_argument("--verify", action="store_true",
                        help="仅验证, 不修改文件")
    parser.add_argument("--dry-run", action="store_true",
                        help="打印计划, 不实际写入")
    args = parser.parse_args()

    if args.verify:
        ok = verify_datasets()
        sys.exit(0 if ok else 1)
    else:
        ok = run_transformation(dry_run=args.dry_run)
        if ok and not args.dry_run:
            print("\n转换完成! 运行 --verify 确认结果.")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
