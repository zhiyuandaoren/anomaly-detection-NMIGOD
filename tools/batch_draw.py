#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量绘图工具 - 遍历所有算法 × 数据集，批量生成对比图
============================================================
用法:
    python tools/batch_draw.py                           # 交互模式
    python tools/batch_draw.py --no-interactive           # 自动扫描全部
    python tools/batch_draw.py --dataset iris,wine       # 仅绘制指定数据集
    python tools/batch_draw.py --algo GCOD,ADFNR          # 仅绘制指定算法
    python tools/batch_draw.py --mode roc                 # 仅绘制 ROC 曲线

输出目录结构:
    images/
    ├── per_dataset/          # 每个数据集一份对比图（所有算法在同一图上）
    │   ├── adult_precision_curve.png
    │   ├── adult_recall_curve.png
    │   ├── adult_f1_curve.png
    │   ├── adult_roc_curve.png
    │   └── ...
    ├── per_algo/             # 每个算法一份汇总图（所有数据集在同一图上）
    │   ├── GCOD_roc_curve.png
    │   └── ...
    └── summary/              # 全局汇总图
        ├── all_f1_curve.png
        └── all_roc_curve.png
"""

import os
import sys
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

try:
    matplotlib.use('Agg')  # 无头模式，不弹窗
except Exception:
    pass

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 字体配置
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 项目配置
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_ROOT = PROJECT_ROOT / 'images'

# ==========================================
# 固定配色方案 — 每个算法颜色唯一, NMIGOD 固定红色
# ==========================================
FIXED_COLORS = {
    'ADFNR':   '#1f77b4',  # 蓝色
    'DASOD':   '#ff7f0e',  # 橙色
    'GCN':     '#2ca02c',  # 绿色
    'GCOD':    '#9467bd',  # 紫色
    'IE':      '#8c564b',  # 棕色
    'KNN':     '#e377c2',  # 粉色
    'NIEOD':   '#17becf',  # 青色
    'NMIGOD':  '#E31818',  # 红色
}
# 后备颜色 (新算法且不在映射表中时使用)
FALLBACK_COLORS = [
    '#bcbd22', '#7f7f7f', '#d62728', '#aec7e8',
    '#ffbb78', '#98df8a', '#c5b0d5', '#c49c94',
    '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]


def get_algo_color(algo_name, _color_idx=0, _algo_list=None):
    """从固定配色表取颜色, NMIGOD=红色, 未知算法用后备色"""
    if algo_name in FIXED_COLORS:
        return FIXED_COLORS[algo_name]
    # 后备: 按字母序分配颜色
    idx = hash(algo_name) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def find_all_results():
    """
    扫描所有 {algo}/output_{dataset}/detection_results.csv
    返回: {algo_name: {dataset_name: csv_path}}
    """
    results = {}
    for item in sorted(PROJECT_ROOT.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith('.') or item.name.startswith('_'):
            continue

        outputs = list(item.glob('output_*/detection_results.csv'))
        if not outputs:
            continue

        algo_results = {}
        for p in outputs:
            ds_name = p.parent.name.replace('output_', '', 1)
            algo_results[ds_name] = str(p)

        if algo_results:
            results[item.name] = algo_results

    return results


def load_algo_data(csv_path):
    """读取 detection_results.csv，返回按分数降序排列的 (scores, actuals)"""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 4:
        return None, None

    raw_scores = df.iloc[:, 1].astype(float).values
    raw_actuals = df.iloc[:, 3].astype(float).values

    sorted_indices = np.argsort(-raw_scores)
    scores = raw_scores[sorted_indices]
    actuals = (raw_actuals[sorted_indices] > 0).astype(int)

    return scores, actuals


def compute_metrics(scores, actuals):
    """向量化计算逐 k 的 Precision/Recall/F1 — O(n) cumsum 加速"""
    n = len(scores)
    total_anomalies = int(actuals.sum())
    if total_anomalies == 0:
        return [0] * n, [0] * n, [0] * n

    # TP(k) = 前k个中有多少真实异常 (scores已降序排列)
    tp_cumsum = np.cumsum(actuals)

    k_vals = np.arange(1, n + 1, dtype=np.float64)

    prec = tp_cumsum / k_vals
    rec = tp_cumsum / total_anomalies
    denom = prec + rec
    f1_val = np.where(denom > 0, 2 * prec * rec / denom, 0.0)

    return prec.tolist(), rec.tolist(), f1_val.tolist()


def draw_per_dataset(all_results, datasets, algos, mode='all'):
    """
    每个数据集一张图，图上包含所有算法的曲线。
    输出到 images/per_dataset/ (SVG 矢量图)
    注: NMIGOD 固定红色
    """
    out_dir = IMAGES_ROOT / 'per_dataset'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 排序: NMIGOD 放最后以便图例醒目
    sorted_algos = sorted(algos, key=lambda a: (0 if a != 'NMIGOD' else 1, a))

    for ds_name in datasets:
        ds_data = {}
        for algo in algos:
            if ds_name in all_results.get(algo, {}):
                scores, actuals = load_algo_data(all_results[algo][ds_name])
                if scores is not None:
                    ds_data[algo] = (scores, actuals)

        if not ds_data:
            continue

        metrics_dict = {}
        for algo_name, (scores, actuals) in ds_data.items():
            prec, rec, f1_val = compute_metrics(scores, actuals)
            metrics_dict[algo_name] = {
                'precision': prec, 'recall': rec, 'f1': f1_val
            }

        n_algo = len(ds_data)
        print(f"  [{ds_name}] {n_algo} 个算法", end="")

        # --- Precision / Recall / F1 曲线 (SVG) ---
        if mode in ('all', 'metrics'):
            for key, label, fname in [
                ('precision', 'Precision', f'{ds_name}_precision_curve.svg'),
                ('recall', 'Recall', f'{ds_name}_recall_curve.svg'),
                ('f1', 'F1-Score', f'{ds_name}_f1_curve.svg'),
            ]:
                fig, ax = plt.subplots(figsize=(10, 6))

                for i, algo_name in enumerate(sorted_algos):
                    if algo_name not in ds_data:
                        continue
                    data = metrics_dict[algo_name][key]
                    k_vals = range(1, len(data) + 1)
                    color = get_algo_color(algo_name, i, sorted_algos)
                    lw = 2.5 if algo_name == 'NMIGOD' else 2
                    ax.plot(k_vals, data, label=algo_name,
                            color=color, linewidth=lw,
                            marker='o', markersize=3,
                            markevery=max(1, len(data) // 40))

                ax.set_xlabel('k (Threshold Index)', fontsize=12, fontweight='bold')
                ax.set_ylabel(label, fontsize=12, fontweight='bold')
                ax.set_title(f'{label} vs k - {ds_name}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=9, loc='best', frameon=True)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_xlim(0, max(len(metrics_dict[a][key]) for a in ds_data))
                ax.set_ylim(0, 1.05)
                fig.tight_layout()

                fig.savefig(out_dir / fname, format='svg', bbox_inches='tight')
                plt.close(fig)

        # --- ROC 曲线 (SVG) ---
        if mode in ('all', 'roc'):
            fig, ax = plt.subplots(figsize=(10, 8))

            for i, algo_name in enumerate(sorted_algos):
                if algo_name not in ds_data:
                    continue
                scores, actuals = ds_data[algo_name]
                fpr, tpr, _ = roc_curve(actuals, scores)
                roc_auc = auc(fpr, tpr)
                color = get_algo_color(algo_name, i, sorted_algos)
                lw = 2.5 if algo_name == 'NMIGOD' else 2
                ax.plot(fpr, tpr, label=f'{algo_name} (AUC={roc_auc:.4f})',
                        color=color, linewidth=lw)

            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title(f'ROC Curve - {ds_name}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=8, loc='lower right', frameon=True)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            fig.tight_layout()

            fig.savefig(out_dir / f'{ds_name}_roc_curve.svg', format='svg', bbox_inches='tight')
            plt.close(fig)

        print(" OK")


def draw_per_algo(all_results, datasets, algos, mode='all'):
    """
    每个算法一张 ROC 汇总图，包含该算法在所有数据集上的曲线。
    输出到 images/per_algo/ (SVG 矢量图)
    """
    out_dir = IMAGES_ROOT / 'per_algo'
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(datasets), 1)))

    for algo in algos:
        fig, ax = plt.subplots(figsize=(12, 8))

        valid_count = 0
        for i, ds_name in enumerate(datasets):
            if ds_name in all_results.get(algo, {}):
                scores, actuals = load_algo_data(all_results[algo][ds_name])
                if scores is not None:
                    fpr, tpr, _ = roc_curve(actuals, scores)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{ds_name} (AUC={roc_auc:.4f})',
                            color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)
                    valid_count += 1

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'ROC Curves - {algo} ({valid_count} datasets)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right', frameon=True, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        fig.tight_layout()

        fig.savefig(out_dir / f'{algo}_roc_curve.svg', format='svg', bbox_inches='tight')
        plt.close(fig)

    print(f"\n[per_algo] 已生成 {len(algos)} 个算法 ROC 汇总图")


def draw_summary(all_results, datasets, algos, mode='all'):
    """
    全局汇总：一张图上比较所有算法的平均性能。
    输出到 images/summary/ (SVG 矢量图)
    注: NMIGOD 固定红色
    """
    out_dir = IMAGES_ROOT / 'summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 排序: NMIGOD 放最后
    sorted_algos = sorted(algos, key=lambda a: (0 if a != 'NMIGOD' else 1, a))

    # 收集所有算法的所有 fpr/tpr（用于平均 ROC）
    if mode in ('all', 'roc'):
        fig, ax = plt.subplots(figsize=(12, 9))

        for i, algo in enumerate(sorted_algos):
            all_fprs = []
            all_tprs = []
            mean_auc = 0
            count = 0

            for ds_name in datasets:
                if ds_name in all_results.get(algo, {}):
                    scores, actuals = load_algo_data(all_results[algo][ds_name])
                    if scores is not None:
                        fpr, tpr, _ = roc_curve(actuals, scores)
                        interp_fpr = np.linspace(0, 1, 100)
                        interp_tpr = np.interp(interp_fpr, fpr, tpr)
                        all_fprs.append(interp_fpr)
                        all_tprs.append(interp_tpr)
                        mean_auc += auc(fpr, tpr)
                        count += 1

            if count > 0:
                mean_tpr = np.mean(all_tprs, axis=0)
                mean_auc /= count
                color = get_algo_color(algo, i, sorted_algos)
                lw = 3.0 if algo == 'NMIGOD' else 2.5
                ax.plot(all_fprs[0], mean_tpr, label=f'{algo} (avg AUC={mean_auc:.4f}, {count} ds)',
                        color=color, linewidth=lw)
                if count > 1:
                    std_tpr = np.std(all_tprs, axis=0)
                    ax.fill_between(all_fprs[0],
                                    np.clip(mean_tpr - std_tpr, 0, 1),
                                    np.clip(mean_tpr + std_tpr, 0, 1),
                                    color=color, alpha=0.15)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Average ROC Curve - All Algorithms', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        fig.tight_layout()

        fig.savefig(out_dir / 'all_roc_curve.svg', format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"[summary] 已生成全局 ROC 汇总图 (SVG)")

    # 收集所有算法的 F1 平均值
    if mode in ('all', 'metrics'):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, algo in enumerate(sorted_algos):
            all_f1 = []
            count = 0
            for ds_name in datasets:
                if ds_name in all_results.get(algo, {}):
                    scores, actuals = load_algo_data(all_results[algo][ds_name])
                    if scores is not None:
                        _, _, f1_list = compute_metrics(scores, actuals)
                        all_f1.append(f1_list)
                        count += 1

            if count > 0:
                max_len = max(len(f) for f in all_f1)
                padded = [f + [f[-1]] * (max_len - len(f)) for f in all_f1]
                mean_f1 = np.mean(padded, axis=0)
                k_vals = range(1, len(mean_f1) + 1)
                color = get_algo_color(algo, i, sorted_algos)
                lw = 3.0 if algo == 'NMIGOD' else 2
                ax.plot(k_vals, mean_f1, label=f'{algo} (avg over {count} ds)',
                        color=color, linewidth=lw)

        ax.set_xlabel('k (Threshold Index)', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Average F1-Score vs k - All Algorithms', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()

        fig.savefig(out_dir / 'all_f1_curve.svg', format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"[summary] 已生成全局 F1 汇总图 (SVG)")


def get_dataset_order(datasets):
    """尝试按数据集大小排序（从对象数少的到多的）"""
    sizes = {}
    # 从第一个有该数据集的算法获取大小
    for ds in datasets:
        for item in PROJECT_ROOT.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                csv_path = item / f'output_{ds}' / 'detection_results.csv'
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        sizes[ds] = len(df)
                    except Exception:
                        pass
                    break
    return sorted(datasets, key=lambda d: sizes.get(d, 0))


def main():
    parser = argparse.ArgumentParser(description='批量绘图工具')
    parser.add_argument('--dataset', type=str, default=None,
                        help='仅绘制指定数据集 (逗号分隔)')
    parser.add_argument('--algo', type=str, default=None,
                        help='仅绘制指定算法 (逗号分隔)')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'metrics', 'roc'],
                        help='绘图模式: all=全部, metrics=P/R/F1曲线, roc=ROC曲线')
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'per_dataset', 'per_algo', 'summary'],
                        help='图表类型: all=全部, per_dataset=数据集对比, per_algo=算法汇总, summary=全局汇总')
    parser.add_argument('--no-interactive', '-n', action='store_true',
                        help='非交互模式')
    args = parser.parse_args()

    # 扫描所有结果
    all_results = find_all_results()

    if not all_results:
        print("未找到任何 detection_results.csv 文件。")
        print("目录结构应为: {算法名}/output_{数据集}/detection_results.csv")
        return

    # 收集所有数据集和算法
    all_datasets_set = set()
    for algo_results in all_results.values():
        all_datasets_set.update(algo_results.keys())
    all_datasets = get_dataset_order(list(all_datasets_set))
    all_algos = sorted(all_results.keys())

    # 过滤
    if args.dataset:
        filter_ds = set(d.strip() for d in args.dataset.split(','))
        datasets = sorted(filter_ds & all_datasets_set, key=lambda d: all_datasets.index(d) if d in all_datasets else 0)
    else:
        datasets = all_datasets

    if args.algo:
        filter_algo = set(a.strip() for a in args.algo.split(','))
        algos = sorted(filter_algo & set(all_algos))
    else:
        algos = all_algos

    if not datasets:
        print("未找到匹配的数据集。")
        return
    if not algos:
        print("未找到匹配的算法。")
        return

    # 输出计划
    print(f"{'=' * 60}")
    print(f"  批量绘图")
    print(f"  数据集: {len(datasets)} 个")
    print(f"  算法:   {len(algos)} 个")
    print(f"  模式:   {args.mode}")
    print(f"  类型:   {args.type}")
    print(f"  输出:   {IMAGES_ROOT}")
    print(f"{'=' * 60}")

    chart_type = args.type
    plot_mode = args.mode

    # 1. 每个数据集对比图
    if chart_type in ('all', 'per_dataset'):
        print(f"\n--- 1/3 数据集对比图 (per_dataset) ---")
        draw_per_dataset(all_results, datasets, algos, mode=plot_mode)
        print(f"  已保存至: {IMAGES_ROOT / 'per_dataset'}")

    # 2. 每个算法汇总图
    if chart_type in ('all', 'per_algo'):
        print(f"\n--- 2/3 算法汇总图 (per_algo) ---")
        draw_per_algo(all_results, datasets, algos, mode=plot_mode)
        print(f"  已保存至: {IMAGES_ROOT / 'per_algo'}")

    # 3. 全局汇总图
    if chart_type in ('all', 'summary'):
        print(f"\n--- 3/3 全局汇总图 (summary) ---")
        draw_summary(all_results, datasets, algos, mode=plot_mode)
        print(f"  已保存至: {IMAGES_ROOT / 'summary'}")

    # 统计
    total_svg = 0
    for d in IMAGES_ROOT.rglob('*.svg'):
        total_svg += 1
    print(f"\n{'=' * 60}")
    print(f"  全部完成! 共生成 {total_svg} 张矢量图 (SVG)")
    print(f"  输出目录: {IMAGES_ROOT}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
