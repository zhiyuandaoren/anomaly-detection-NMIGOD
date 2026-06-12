#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测结果汇总工具
============================================================
扫描指定目录下所有算法的 metrics.csv 文件，统计精确率、召回率、
F1分数、AUC值，生成对比表格。

用法:
    python tools/collect_metrics.py                         # 交互模式
    python tools/collect_metrics.py --base ./               # 指定扫描目录
    python tools/collect_metrics.py --output summary.csv    # 指定输出文件
    python tools/collect_metrics.py --list                  # 仅列出可扫描的目录, 不生成表格
    python tools/collect_metrics.py --best                  # 每个数据集标注最佳算法

输出表格格式:
    第一列: 数据集名称
    后续列: 每个算法的 Precision, Recall, F1-Score, AUC  (多级列名)
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def find_algorithm_dirs(base_path):
    """
    扫描 base_path 下所有包含 output_*/metrics.csv 的一级子目录。
    返回: {算法目录名: [数据集子目录列表]}
    """
    base = Path(base_path).resolve()
    algo_dirs = {}

    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith('.') or item.name.startswith('_'):
            continue

        outputs = sorted(item.glob('output_*/metrics.csv'))
        if outputs:
            # 提取数据集名: output_{dataset}/metrics.csv -> dataset
            datasets = []
            for p in outputs:
                ds_name = p.parent.name.replace('output_', '', 1)
                datasets.append((ds_name, str(p)))
            if datasets:
                algo_dirs[item.name] = datasets

    return algo_dirs


def read_metrics(csv_path):
    """读取单个 metrics.csv, 返回 {Metric: Value} 字典"""
    try:
        df = pd.read_csv(csv_path)
        metrics = {}
        for _, row in df.iterrows():
            metrics[str(row['Metric']).strip()] = row['Value']
        return metrics
    except Exception as e:
        print(f"  [WARN] 读取失败 {csv_path}: {e}")
        return {}


def collect_all_metrics(algo_dirs):
    """
    收集所有算法的所有数据集指标。

    返回: list of dicts, 每个 dict 包含:
        { 'Dataset': 数据集名,
          '{algo}_Precision': val, '{algo}_Recall': val,
          '{algo}_F1-Score': val, '{algo}_AUC': val }
    """
    # 收集所有数据集名 (并集)
    all_datasets = set()
    for algo, datasets in algo_dirs.items():
        for ds_name, _ in datasets:
            all_datasets.add(ds_name)
    all_datasets = sorted(all_datasets)

    rows = []
    for ds in all_datasets:
        row = {'Dataset': ds}
        for algo, datasets in algo_dirs.items():
            ds_dict = dict(datasets)
            if ds in ds_dict:
                m = read_metrics(ds_dict[ds])
                row[f'{algo}_Precision'] = m.get('Precision', None)
                row[f'{algo}_Recall']    = m.get('Recall', None)
                row[f'{algo}_F1-Score']  = m.get('F1-Score', None)
                row[f'{algo}_AUC']       = m.get('AUC', None)
            else:
                for metric in ['Precision', 'Recall', 'F1-Score', 'AUC']:
                    row[f'{algo}_{metric}'] = None
        rows.append(row)
    return rows


def build_multiindex_df(rows, algos):
    """构建带多级列名的 DataFrame"""
    data = {}
    for row in rows:
        ds = row['Dataset']
        for algo in algos:
            for metric in ['Precision', 'Recall', 'F1-Score', 'AUC']:
                col = f'{algo}_{metric}'
                if ds not in data:
                    data[ds] = {}
                data[ds][col] = row.get(col, None)

    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Dataset'

    # 构建多级列名: (算法, 指标)
    columns = []
    for algo in algos:
        for metric in ['Precision', 'Recall', 'F1-Score', 'AUC']:
            columns.append((algo, metric))
    # 为存在的列构建 MultiIndex
    existing_cols = [(a, m) for a, m in columns if f'{a}_{m}' in df.columns]
    df = df[[f'{a}_{m}' for a, m in existing_cols]]
    df.columns = pd.MultiIndex.from_tuples(existing_cols)
    return df


def highlight_best(df):
    """在每个数据集行中, 对每个指标标注最佳算法"""
    metrics = ['Precision', 'Recall', 'F1-Score', 'AUC']
    styled = df.copy()

    print("\n--- 各指标最佳算法 ---")
    for metric in metrics:
        best_algos = {}
        for ds in df.index:
            vals = {}
            for col in df.columns:
                if col[1] == metric:
                    vals[col[0]] = df.loc[ds, col]
            if vals:
                # 过滤 NaN
                valid = {k: v for k, v in vals.items() if not pd.isna(v)}
                if valid:
                    best = max(valid, key=valid.get)
                    best_algos[ds] = (best, valid[best])

        # 统计各算法赢得"最佳"的次数
        win_count = defaultdict(int)
        for ds, (algo, val) in best_algos.items():
            win_count[algo] += 1

        print(f"\n  {metric}:")
        for algo, count in sorted(win_count.items(), key=lambda x: -x[1]):
            print(f"    {algo}: {count} 个数据集最佳")

    return styled


def save_separate_tables(rows, algos, output_dir):
    """为每个指标生成独立的 CSV 文件，Dataset 为行, 算法为列"""
    metrics = ['Precision', 'Recall', 'F1-Score', 'AUC']
    filenames = ['precision.csv', 'recall.csv', 'f1_score.csv', 'auc.csv']
    base_path = Path(output_dir)

    for metric, fname in zip(metrics, filenames):
        data = {}
        for row in rows:
            ds = row['Dataset']
            data[ds] = {}
            for algo in algos:
                val = row.get(f'{algo}_{metric}')
                data[ds][algo] = round(val, 4) if val is not None else None

        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.name = 'Dataset'
        df = df[algos]  # 保持算法顺序一致

        out_path = base_path / fname
        df.to_csv(out_path, encoding='utf-8-sig')
        print(f"  [{metric}] -> {out_path}")

    return True


def print_compact_table(rows, algos):
    """打印紧凑表格到终端"""
    metrics = ['Precision', 'Recall', 'F1-Score', 'AUC']

    # 表头
    header = f"{'Dataset':<22s}"
    for algo in algos:
        header += f" | {algo:<10s}"
    print(header)
    print("-" * len(header))

    for metric in metrics:
        print(f"\n  [{metric}]")
        for row in rows:
            ds = row['Dataset']
            line = f"  {ds:<20s}"
            for algo in algos:
                val = row.get(f'{algo}_{metric}')
                if val is not None:
                    line += f" | {val:<10.4f}"
                else:
                    line += f" | {'-':<10s}"
            print(line)


def interactive_select_algos(algo_dirs):
    """交互式选择要包含的算法目录"""
    print("\n扫描到以下包含结果数据的目录:\n")
    for i, (dname, datasets) in enumerate(algo_dirs.items(), 1):
        n_ds = len(set(d for d, _ in datasets))
        n_csv = len(datasets)
        print(f"  [{i}] {dname}  ({n_ds} 个数据集, {n_csv} 个 metrics.csv)")

    print(f"\n  [A] 全部选择")
    print(f"  [Q] 退出")

    choice = input("\n请选择 (数字/字母, 逗号分隔, 默认A): ").strip().upper()

    if not choice or choice == 'A':
        return {k: k for k in algo_dirs.keys()}

    if choice == 'Q':
        return None

    selected = {}
    parts = [p.strip() for p in choice.split(',')]
    dir_list = list(algo_dirs.keys())

    for p in parts:
        if p.isdigit():
            idx = int(p) - 1
            if 0 <= idx < len(dir_list):
                dname = dir_list[idx]
                label = input(f"  为 '{dname}' 输入算法名称 (回车使用默认): ").strip()
                selected[dname] = label if label else dname
        elif p in algo_dirs:
            label = input(f"  为 '{p}' 输入算法名称 (回车使用默认): ").strip()
            selected[p] = label if label else p

    return selected


def interactive_main(base_path):
    """全交互式模式: 通过终端问答收集所有参数并执行汇总"""
    print("\n" + "=" * 60)
    print("  异常检测结果汇总工具 (交互式模式)")
    print("=" * 60)

    # 步骤1: 扫描目录
    print(f"\n当前扫描目录: {base_path}")
    change = input("是否更换扫描目录? (y/n, 默认n): ").strip().lower()
    if change == 'y':
        new_path = input("请输入新的扫描目录路径: ").strip()
        if new_path and os.path.isdir(new_path):
            base_path = os.path.abspath(new_path)
        else:
            print(f"[WARN] 目录无效, 使用默认: {base_path}")

    algo_dirs = find_algorithm_dirs(base_path)
    if not algo_dirs:
        print("未找到任何包含 metrics.csv 的算法目录。")
        print("确保目录结构为: {算法名}/output_{数据集}/metrics.csv")
        return

    # 步骤2: 选择算法
    selected = interactive_select_algos(algo_dirs)
    if selected is None:
        print("已取消。")
        return
    if not selected:
        print("未选择任何算法目录。")
        return

    # 步骤3: 选择输出选项
    print("\n--- 输出选项 ---")

    # 输出文件
    default_output = os.path.join(base_path, 'metrics_summary.csv')
    out_choice = input(f"输出CSV文件路径 (回车使用默认: {default_output}): ").strip()
    output_path = out_choice if out_choice else default_output

    # 是否生成分指标独立表格
    split_choice = input("是否为每个指标生成独立的CSV文件 (Precision, Recall, F1-Score, AUC)? (y/n, 默认y): ").strip().lower()
    do_split = split_choice != 'n'

    # 是否紧凑打印
    compact_choice = input("是否紧凑打印全部指标到终端? (y/n, 默认y): ").strip().lower()
    do_compact = compact_choice != 'n'

    # 是否标注最佳算法
    best_choice = input("是否统计各指标最佳算法? (y/n, 默认y): ").strip().lower()
    do_best = best_choice != 'n'

    # 筛选后重新收集
    filtered_dirs = {k: algo_dirs[k] for k in selected.keys() if k in algo_dirs}
    renamed_dirs = {}
    for orig_name, datasets in filtered_dirs.items():
        new_name = selected[orig_name]
        renamed_dirs[new_name] = datasets

    # 收集指标
    print("\n正在读取 metrics.csv 文件...")
    rows = collect_all_metrics(renamed_dirs)

    if not rows:
        print("未收集到任何数据。")
        return

    algos = sorted(renamed_dirs.keys())
    ds_count = len(rows)
    print(f"收集完成: {ds_count} 个数据集, {len(algos)} 个算法")

    # 构建 DataFrame
    df = build_multiindex_df(rows, algos)

    # 输出文件
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, encoding='utf-8-sig')
    print(f"\n完整多级表格已保存至: {output_path}")

    # 分别生成独立的指标表
    if do_split:
        print(f"\n分指标独立表格:")
        save_separate_tables(rows, algos, os.path.dirname(output_path) or base_path)

    # 紧凑终端打印
    if do_compact:
        print_compact_table(rows, algos)

    # 最佳算法标注
    if do_best:
        highlight_best(df)

    # 默认打印摘要
    if not do_compact:
        print(f"\n--- 指标摘要 (F1-Score) ---")
        print(f"{'Dataset':<22s}", end='')
        for algo in algos:
            print(f" | {algo:>10s}", end='')
        print()
        print('-' * (22 + 13 * len(algos)))

        for row in rows:
            ds = row['Dataset']
            print(f"{ds:<22s}", end='')
            for algo in algos:
                val = row.get(f'{algo}_F1-Score')
                if val is not None:
                    print(f" | {val:>10.4f}", end='')
                else:
                    print(f" | {'-':>10s}", end='')
            print()

        # 平均指标
        print('-' * (22 + 13 * len(algos)))
        print(f"{'[平均]':<22s}", end='')
        for algo in algos:
            vals = [row[f'{algo}_F1-Score'] for row in rows
                    if row.get(f'{algo}_F1-Score') is not None]
            if vals:
                print(f" | {np.mean(vals):>10.4f}", end='')
            else:
                print(f" | {'-':>10s}", end='')
        print()


def main():
    parser = argparse.ArgumentParser(description='异常检测结果汇总工具')
    parser.add_argument('--base', '-b', type=str, default=None,
                        help='扫描根目录 (默认: 当前目录)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出CSV文件路径 (默认: metrics_summary.csv)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='仅列出可扫描的目录, 不生成表格')
    parser.add_argument('--best', action='store_true',
                        help='标注各指标最佳算法')
    parser.add_argument('--compact', '-c', action='store_true',
                        help='紧凑输出 (每个指标单独一行)')
    parser.add_argument('--split', '-s', action='store_true',
                        help='分别为每个指标生成独立的CSV文件 (Precision, Recall, F1-Score, AUC)')
    parser.add_argument('--no-interactive', '-n', action='store_true',
                        help='非交互模式, 自动选择所有目录, 目录名作为算法名')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='强制使用全交互式模式')
    args = parser.parse_args()

    base_path = args.base or os.getcwd()
    base_path = os.path.abspath(base_path)

    if not os.path.isdir(base_path):
        print(f"[ERROR] 目录不存在: {base_path}")
        return

    # 检查是否有命令行参数 (非交互模式)
    has_cli_args = any([
        args.list, args.best, args.compact, args.split, args.no_interactive,
        args.output is not None, args.base is not None
    ])

    # 交互式模式: 无参数或显式指定 --interactive
    if args.interactive or not has_cli_args:
        # 先扫描查看结果
        algo_dirs = find_algorithm_dirs(base_path)
        if not algo_dirs:
            print("未找到任何包含 metrics.csv 的算法目录。")
            print("确保目录结构为: {算法名}/output_{数据集}/metrics.csv")
            return

        if args.list:
            print(f"\n找到 {len(algo_dirs)} 个算法目录:\n")
            for dname, datasets in algo_dirs.items():
                ds_names = sorted(set(d for d, _ in datasets))
                print(f"  {dname}/  ({len(ds_names)} 个数据集)")
                for ds in ds_names:
                    print(f"    - output_{ds}/")
            return

        interactive_main(base_path)
        return

    # 命令行模式 (原有逻辑)
    print(f"扫描目录: {base_path}")
    algo_dirs = find_algorithm_dirs(base_path)

    if not algo_dirs:
        print("未找到任何包含 metrics.csv 的算法目录。")
        print("确保目录结构为: {算法名}/output_{数据集}/metrics.csv")
        return

    if args.list:
        print(f"\n找到 {len(algo_dirs)} 个算法目录:\n")
        for dname, datasets in algo_dirs.items():
            ds_names = sorted(set(d for d, _ in datasets))
            print(f"  {dname}/  ({len(ds_names)} 个数据集)")
            for ds in ds_names:
                print(f"    - output_{ds}/")
        return

    # 选择算法
    if args.no_interactive:
        selected = {k: k for k in algo_dirs.keys()}
    else:
        selected = interactive_select_algos(algo_dirs)
        if selected is None:
            print("已取消。")
            return
        if not selected:
            print("未选择任何算法目录。")
            return

    # 筛选后重新收集
    filtered_dirs = {k: algo_dirs[k] for k in selected.keys() if k in algo_dirs}
    renamed_dirs = {}
    for orig_name, datasets in filtered_dirs.items():
        new_name = selected[orig_name]
        renamed_dirs[new_name] = datasets

    # 收集指标
    print("\n正在读取 metrics.csv 文件...")
    rows = collect_all_metrics(renamed_dirs)

    if not rows:
        print("未收集到任何数据。")
        return

    algos = sorted(renamed_dirs.keys())
    ds_count = len(rows)

    print(f"收集完成: {ds_count} 个数据集, {len(algos)} 个算法")

    # 构建 DataFrame
    df = build_multiindex_df(rows, algos)

    # 输出文件
    output_path = args.output or os.path.join(base_path, 'metrics_summary.csv')
    df.to_csv(output_path, encoding='utf-8-sig')
    print(f"\n完整多级表格已保存至: {output_path}")

    # 分别生成独立的指标表
    if args.split:
        print(f"\n分指标独立表格:")
        save_separate_tables(rows, algos, os.path.dirname(output_path) or base_path)

    # 紧凑终端打印
    if args.compact:
        print_compact_table(rows, algos)

    # 最佳算法标注
    if args.best:
        highlight_best(df)

    # 默认打印摘要
    if not args.compact:
        print(f"\n--- 指标摘要 (F1-Score) ---")
        print(f"{'Dataset':<22s}", end='')
        for algo in algos:
            print(f" | {algo:>10s}", end='')
        print()
        print('-' * (22 + 13 * len(algos)))

        for row in rows:
            ds = row['Dataset']
            print(f"{ds:<22s}", end='')
            for algo in algos:
                val = row.get(f'{algo}_F1-Score')
                if val is not None:
                    print(f" | {val:>10.4f}", end='')
                else:
                    print(f" | {'-':>10s}", end='')
            print()

        # 平均指标
        print('-' * (22 + 13 * len(algos)))
        print(f"{'[平均]':<22s}", end='')
        for algo in algos:
            vals = [row[f'{algo}_F1-Score'] for row in rows
                    if row.get(f'{algo}_F1-Score') is not None]
            if vals:
                print(f" | {np.mean(vals):>10.4f}", end='')
            else:
                print(f" | {'-':>10s}", end='')
        print()


if __name__ == '__main__':
    main()