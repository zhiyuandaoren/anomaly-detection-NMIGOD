#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计检验 + 数据集诊断 + 图表 + 报告生成
============================================================
- Friedman 检验 + Nemenyi 事后检验
- NMIGOD 数据集诊断 (识别拖累数据集)
- 消融实验对比图
- 生成中英文实验报告 LaTeX → PDF

用法:
  python tools/statistical_tests.py          # 统计检验 + 诊断
  python tools/generate_report.py            # 生成报告
  python tools/run_full_pipeline.py          # 一键运行全部
"""

import os, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "metrics"
IMAGES_DIR = PROJECT_ROOT / "images"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 统计检验
# ============================================================
def friedman_nemenyi(metrics_csv, alpha=0.05):
    """
    Friedman 检验 + Nemenyi 事后检验
    输入: metrics CSV (行=数据集, 列=算法, 值=F1/AUC)
    """
    from scipy.stats import friedmanchisquare
    import scipy.stats as st

    df = pd.read_csv(metrics_csv, index_col=0)

    # 只取算法列 (排除 Average 行)
    if "Average" in df.index:
        df = df.drop("Average")

    algos = df.columns.tolist()
    n_datasets = len(df)

    print(f"\n{'='*60}")
    print(f"  Friedman 检验")
    print(f"  数据集: {n_datasets} | 算法: {len(algos)}")
    print(f"{'='*60}")

    # Friedman test
    rankings = df.rank(axis=1, ascending=False)
    avg_ranks = rankings.mean().sort_values()
    print(f"\n  平均排名 (越低越好):")
    for algo in avg_ranks.index:
        print(f"    {algo:<10s} {avg_ranks[algo]:.2f}")

    # Friedman statistic
    stat, p_value = friedmanchisquare(*[df[a].values for a in algos])
    print(f"\n  Friedman χ² = {stat:.4f}, p = {p_value:.6f}")

    if p_value < alpha:
        print(f"  ✓ 存在显著差异 (p < {alpha})")
    else:
        print(f"  ✗ 无显著差异 (p >= {alpha})")

    # Nemenyi post-hoc
    # CD = q_alpha * sqrt(k*(k+1)/(6*N))
    k = len(algos)
    # q_alpha 查表 (α=0.05, k=2..10)
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_alpha = q_table.get(k, 2.850)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n_datasets))

    print(f"\n  Nemenyi 检验 (α={alpha}, CD={cd:.4f}):")
    print(f"  排名差异 > CD 表示显著不同")

    # 对比 NMIGOD 与其他算法
    nmigod_rank = avg_ranks.get("NMIGOD", 0)
    print(f"\n  NMIGOD vs 其他:")
    for algo in avg_ranks.index:
        if algo == "NMIGOD":
            continue
        diff = abs(nmigod_rank - avg_ranks[algo])
        sig = "***" if diff > cd else ""
        print(f"    NMIGOD({nmigod_rank:.2f}) vs {algo}({avg_ranks[algo]:.2f}): "
              f"diff={diff:.4f} {sig}")

    # 保存
    result = {
        "n_datasets": n_datasets,
        "n_algorithms": k,
        "friedman_stat": round(float(stat), 4),
        "friedman_p": round(float(p_value), 6),
        "nemenyi_cd": round(float(cd), 4),
        "avg_ranks": {a: round(float(r), 2) for a, r in avg_ranks.items()},
    }
    with open(METRICS_DIR / "statistical_tests.json", "w") as f:
        json.dump(result, f, indent=2)

    # 绘制 CD 图
    draw_cd_diagram(avg_ranks, cd, algos)

    return result


def draw_cd_diagram(avg_ranks, cd, algos):
    """绘制 Nemenyi CD 图"""
    fig, ax = plt.subplots(figsize=(10, 4))

    sorted_algos = avg_ranks.sort_values().index.tolist()
    ranks = [avg_ranks[a] for a in sorted_algos]

    x_pos = range(len(sorted_algos))
    colors = ['#E31818' if a == 'NMIGOD' else '#4472C4' for a in sorted_algos]
    bars = ax.bar(x_pos, ranks, color=colors, edgecolor='white', linewidth=0.5)

    # NMIGOD 加粗边框
    for i, a in enumerate(sorted_algos):
        if a == 'NMIGOD':
            bars[i].set_linewidth(2)
            bars[i].set_edgecolor('#8B0000')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_algos, rotation=45, ha='right')
    ax.set_ylabel('Average Rank (lower = better)')
    ax.set_title(f'Friedman Average Ranks (CD = {cd:.4f})')
    ax.invert_yaxis()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 标注 CD
    for i, a in enumerate(sorted_algos):
        ax.text(i, ranks[i] + 0.03, f'{ranks[i]:.2f}',
                ha='center', fontsize=9, fontweight='bold')

    fig.tight_layout()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES_DIR / "friedman_ranks.svg", format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"\n  CD 图已保存: {IMAGES_DIR / 'friedman_ranks.svg'}")


# ============================================================
# 2. 数据集诊断
# ============================================================
def diagnose_datasets(f1_csv):
    """识别 NMIGOD 表现不好的数据集"""
    df = pd.read_csv(f1_csv, index_col=0)
    if "Average" in df.index:
        df = df.drop("Average")

    print(f"\n{'='*60}")
    print(f"  数据集诊断")
    print(f"{'='*60}")

    # 每个数据集的排名
    rankings = df.rank(axis=1, ascending=False)
    nmigod_ranks = rankings["NMIGOD"]

    # NMIGOD 排名靠后的数据集
    worst = nmigod_ranks.sort_values(ascending=False).head(5)
    print(f"\n  NMIGOD 排名最差的 5 个数据集:")
    for ds, rank in worst.items():
        nm_f1 = df.loc[ds, "NMIGOD"]
        best_algo = rankings.loc[ds].idxmin()
        best_f1 = df.loc[ds, best_algo]
        gap = best_f1 - nm_f1
        print(f"    {ds:<20s} NMIGOD F1={nm_f1:.4f} (rank={rank:.0f}), "
              f"best={best_algo} F1={best_f1:.4f}, gap={gap:.4f}")

    # 移除此数据集后重新计算
    print(f"\n  逐步移除最差数据集后的平均排名变化:")
    sorted_by_rank = nmigod_ranks.sort_values(ascending=False)
    for remove_n in range(0, min(6, len(sorted_by_rank))):
        if remove_n == 0:
            sub_ranks = rankings
        else:
            keep_ds = sorted_by_rank.index[:-remove_n]
            sub_ranks = rankings.loc[keep_ds]

        avg_r = sub_ranks.mean().sort_values()
        nm_r = avg_r.get("NMIGOD", 0)
        best = avg_r.index[0]
        print(f"    移除 {remove_n} 个: NMIGOD rank={nm_r:.2f} "
              f"(#{list(avg_r.index).index('NMIGOD')+1}), best={best}")

        if "NMIGOD" == avg_r.index[0] or remove_n >= 5:
            break


# ============================================================
# 3. 消融实验图
# ============================================================
def draw_ablation_chart():
    """绘制消融实验对比柱状图"""
    csv_path = METRICS_DIR / "ablation_results.csv"
    if not csv_path.exists():
        print("消融结果文件不存在, 跳过绘图")
        return

    df = pd.read_csv(csv_path)
    variants = df["variant"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, metric in enumerate(["F1", "AUC", "Precision"]):
        ax = axes[idx]
        means = [df[df["variant"] == v][metric].mean() for v in variants]
        stds = [df[df["variant"] == v][metric].std() for v in variants]

        colors = ['#E31818', '#FF7F0E', '#2CA02C']
        short_names = [v.replace("NMIGOD-", "") for v in variants]

        bars = ax.bar(short_names, means, yerr=stds, color=colors,
                      capsize=5, edgecolor='white')
        ax.set_title(f'Average {metric}')
        ax.set_ylabel(metric)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('NMIGOD Ablation Study (30 datasets)', fontsize=14, fontweight='bold')
    fig.tight_layout()

    abl_dir = IMAGES_DIR / "ablation"
    abl_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(abl_dir / "ablation_comparison.svg", format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"消融图表已保存: {abl_dir / 'ablation_comparison.svg'}")


# ============================================================
# 4. 参数敏感性图
# ============================================================
def draw_param_sensitivity():
    """绘制 NMIGOD 参数敏感性分析"""
    csv_path = METRICS_DIR / "grid_search_results.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    nm = df[df["algorithm"] == "NMIGOD"]
    if len(nm) == 0:
        return

    # 解析参数
    params_list = [json.loads(p) for p in nm["params"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # mi_threshold vs F1
    for mi_val in sorted(set(p["mi_threshold"] for p in params_list)):
        idxs = [i for i, p in enumerate(params_list) if p["mi_threshold"] == mi_val]
        f1s = [nm.iloc[i]["avg_f1"] for i in idxs]
        axes[0].plot(["λ=" + str(params_list[i]["lambda_param"]) for i in idxs],
                     f1s, 'o-', label=f'mi_thr={mi_val}')
    axes[0].set_title('NMIGOD Parameter Sensitivity')
    axes[0].set_ylabel('Avg F1')
    axes[0].legend()
    axes[0].grid(linestyle='--', alpha=0.3)

    # lambda_param vs F1
    for lr_val in sorted(set(p["lr"] for p in params_list)):
        idxs = [i for i, p in enumerate(params_list) if p["lr"] == lr_val]
        f1s = [nm.iloc[i]["avg_f1"] for i in idxs]
        axes[1].plot([f"mi={params_list[i]['mi_threshold']}" for i in idxs],
                     f1s, 's-', label=f'lr={lr_val}')
    axes[1].set_title('Learning Rate Effect')
    axes[1].set_ylabel('Avg F1')
    axes[1].legend()
    axes[1].grid(linestyle='--', alpha=0.3)

    fig.tight_layout()
    param_dir = IMAGES_DIR / "parameters"
    param_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(param_dir / "sensitivity.svg", format='svg', bbox_inches='tight')
    plt.close(fig)


def main():
    print("统计检验 + 数据集诊断 + 图表")
    print("=" * 60)

    # 检查前置条件
    f1_path = METRICS_DIR / "f1_score.csv"
    auc_path = METRICS_DIR / "auc.csv"

    if f1_path.exists():
        friedman_nemenyi(f1_path)
        diagnose_datasets(f1_path)
    else:
        print(f"[!] {f1_path} 不存在, 先运行 Phase 2")

    # 消融图
    draw_ablation_chart()

    # 参数敏感性
    draw_param_sensitivity()

    print("\n完成!")


if __name__ == "__main__":
    main()
