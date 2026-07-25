#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数网格搜索 — 在 8 个代表性数据集上搜索各算法最佳参数
============================================================
用法:
  python tools/grid_search.py                  # 运行全部算法
  python tools/grid_search.py --algo NMIGOD    # 仅运行指定算法
  python tools/grid_search.py --dry-run        # 仅打印计划
"""

import os, sys, time, gc, json, itertools, warnings
import importlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets"

# GPU 检测
try:
    import torch
    CUDA_OK = torch.cuda.is_available()
except ImportError:
    CUDA_OK = False

# 文件名映射
FILE_NAME_MAP = {
    "breast-cancer": "breast-cancer-wisconsin",
    "wine-red": "winequality-red",
    "wine-white": "winequality-white",
}

def resolve_filename(name):
    return FILE_NAME_MAP.get(name, name)

# ============================================================
# 代表数据集 (8 个)
# ============================================================
REPRESENTATIVE_DATASETS = [
    {"name": "iris",       "outlier_col": "class", "outlier_val": "anomaly"},   # 120 rows
    {"name": "wine",       "outlier_col": "class", "outlier_val": "anomaly"},   # 160 rows
    {"name": "glass",      "outlier_col": "class", "outlier_val": "anomaly"},   # 214 rows
    {"name": "diabetes",   "outlier_col": "class", "outlier_val": "anomaly"},   # 370 rows
]

# ============================================================
# 参数网格定义
# ============================================================
PARAM_GRIDS = {
    "ADFNR": {
        "epsilon": [0.3, 0.5, 0.7],
    },
    "DASOD": {
        "K": [3, 5, 7],
        "lambda_ratio": [0.03, 0.05, 0.10],
    },
    "GCN": {
        "k_neighbors": [10, 15, 20],
        "hidden1": [128], "hidden2": [64],  # 固定架构
        "epochs": [200],
        "lr": [0.001, 0.01],
    },
    "GCN-LOF": {
        "k_neighbors": [10, 15, 20],
        "lof_neighbors": [15, 20, 30],
        "hidden1": [128], "hidden2": [64],
        "epochs": [200],
        "lr": [0.001, 0.01],
    },
    "NIEOD": {
        "lambda_param": [0.5, 1.0, 1.5, 2.0],
    },
    "NMIGOD": {
        "lambda_param": [0.5, 1.0, 1.5],
        "mi_threshold": [0.03, 0.05, 0.10],
        "hidden1": [128], "hidden2": [64],
        "epochs": [200],
        "lr": [0.001, 0.01],
    },
}

ALGO_SCRIPTS = {
    "ADFNR":    "ADFNR/detector.py",
    "DASOD":    "DASOD/detector.py",
    "GCN":      "GCN/detector.py",
    "GCN-LOF":  "GCN-LOF/detector.py",
    "NIEOD":    "NIEOD/detector.py",
    "NMIGOD":   "NMIGOD/detector.py",
}


def import_algorithm_module(algo_name):
    script_path = PROJECT_ROOT / ALGO_SCRIPTS[algo_name]
    module_name = f"_gs_{algo_name.replace('-', '_').lower()}"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.AnomalyDetectionFramework


def run_single(fw, ds_cfg, output_dir, force_cpu=False):
    """运行单个算法-数据集流水线, 返回 metrics dict."""
    os.makedirs(output_dir, exist_ok=True)

    name = ds_cfg["name"]
    actual_name = resolve_filename(name)
    csv_path = DATA_ROOT / f"{actual_name}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    anomaly_vals = [v.strip() for v in ds_cfg["outlier_val"].split(',') if v.strip()]

    fw.df_raw = df
    fw.target_column = ds_cfg["outlier_col"]
    fw.anomaly_values = anomaly_vals
    fw.output_folder = str(output_dir)

    if force_cpu and hasattr(fw, 'device'):
        fw.device = torch.device('cpu')

    try:
        fw.preprocess_data()
        fw.train_model()
        fw.get_anomaly_scores()
        fw.optimize_threshold()
        y_pred = fw.calculate_metrics_and_topk()
        fw.save_results(y_pred)

        # 读取指标
        mpath = os.path.join(output_dir, "metrics.csv")
        if os.path.exists(mpath):
            m = pd.read_csv(mpath)
            return dict(zip(m["Metric"], m["Value"]))
    except Exception as e:
        print(f"    [ERROR] {e}")
    return None


def grid_search(algo_name, datasets, param_grid, force_cpu=False):
    """对指定算法执行网格搜索."""
    print(f"\n{'='*60}")
    print(f"  网格搜索: {algo_name}")
    print(f"  数据集: {len(datasets)} 个 | 参数组合: {count_combos(param_grid)} 个")
    print(f"{'='*60}")

    # 生成所有参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    combo_dicts = [dict(zip(keys, combo)) for combo in combos]

    print(f"  参数组合列表 ({len(combo_dicts)}):")
    for cd in combo_dicts:
        print(f"    {cd}")

    FrameworkClass = import_algorithm_module(algo_name)
    results = []

    for idx, params in enumerate(combo_dicts):
        print(f"\n  [{idx+1}/{len(combo_dicts)}] params={params}")

        f1_scores = []
        auc_scores = []
        precisions = []
        recalls = []

        for ds_cfg in datasets:
            ds_name = ds_cfg["name"]
            out_dir = PROJECT_ROOT / algo_name / f"output_gs_{ds_name}"
            # 清理旧的搜索输出
            import shutil
            if out_dir.exists():
                shutil.rmtree(out_dir)

            try:
                fw = FrameworkClass(**params)
            except TypeError:
                fw = FrameworkClass()

            metrics = run_single(fw, ds_cfg, out_dir)
            if metrics:
                f1_scores.append(metrics.get("F1-Score", 0))
                auc_scores.append(metrics.get("AUC", 0))
                precisions.append(metrics.get("Precision", 0))
                recalls.append(metrics.get("Recall", 0))
                print(f"    {ds_name}: F1={metrics.get('F1-Score',0):.4f} "
                      f"AUC={metrics.get('AUC',0):.4f}")
            else:
                f1_scores.append(0)
                auc_scores.append(0)
                precisions.append(0)
                recalls.append(0)

            # 清理
            del fw
            gc.collect()
            if CUDA_OK:
                torch.cuda.empty_cache()

        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        avg_auc = np.mean(auc_scores) if auc_scores else 0

        results.append({
            "algorithm": algo_name,
            "params": json.dumps(params),
            "avg_f1": round(avg_f1, 4),
            "avg_auc": round(avg_auc, 4),
            "per_dataset_f1": json.dumps(dict(zip(
                [d["name"] for d in datasets], [round(f, 4) for f in f1_scores]))),
        })
        print(f"    => avg F1={avg_f1:.4f}, avg AUC={avg_auc:.4f}")

    # 选最佳
    best = max(results, key=lambda r: r["avg_f1"])
    print(f"\n  [{algo_name}] 最佳参数: {best['params']} (avg F1={best['avg_f1']:.4f})")
    return results, best


def count_combos(param_grid):
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n


def main():
    import argparse
    parser = argparse.ArgumentParser(description="参数网格搜索")
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "metrics" / "best_params.csv"))
    args = parser.parse_args()

    if args.algo:
        algos = {args.algo: PARAM_GRIDS[args.algo]}
    else:
        algos = PARAM_GRIDS

    # 统计
    total_combos = sum(count_combos(g) for g in algos.values())
    total_runs = total_combos * len(REPRESENTATIVE_DATASETS)
    print(f"网格搜索计划:")
    print(f"  算法: {len(algos)} 个")
    print(f"  数据集: {len(REPRESENTATIVE_DATASETS)} 个")
    print(f"  参数组合总数: {total_combos}")
    print(f"  总运行次数: {total_runs}")
    print(f"  GPU: {'可用' if CUDA_OK else '不可用'}")

    if args.dry_run:
        return

    all_best = []
    all_results = []
    log_lines = []
    t0 = time.time()

    for algo_name, grid in algos.items():
        t1 = time.time()
        results, best = grid_search(algo_name, REPRESENTATIVE_DATASETS, grid,
                                    force_cpu=args.cpu)
        all_results.extend(results)
        all_best.append(best)
        elapsed = time.time() - t1

        log_lines.append(f"[{algo_name}] best_params={best['params']} "
                        f"avg_f1={best['avg_f1']:.4f} time={elapsed:.0f}s")

        # 增量保存
        pd.DataFrame(all_results).to_csv(
            PROJECT_ROOT / "metrics" / "grid_search_results.csv", index=False)
        pd.DataFrame(all_best).to_csv(args.output, index=False)

        print(log_lines[-1])

    total_elapsed = time.time() - t0
    print(f"\n总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"\n最佳参数汇总:")
    for b in all_best:
        print(f"  {b['algorithm']:<10s} {b['params']:<40s} F1={b['avg_f1']:.4f}")

    # 保存日志
    log_path = PROJECT_ROOT / "metrics" / "grid_search_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Grid Search Log — {datetime.now()}\n")
        f.write(f"Datasets: {[d['name'] for d in REPRESENTATIVE_DATASETS]}\n\n")
        for l in log_lines:
            f.write(l + "\n")

    print(f"\n最佳参数表: {args.output}")
    print(f"搜索详情: {PROJECT_ROOT / 'metrics' / 'grid_search_results.csv'}")


if __name__ == "__main__":
    main()
