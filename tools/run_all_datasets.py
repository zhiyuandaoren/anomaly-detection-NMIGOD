#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行所有异常检测算法在所有数据集上的检测任务
============================================================
逐算法导入执行, 避免 subprocess 开销, PyTorch 算法自动使用 GPU 加速。

用法: python tools/run_all_datasets.py [选项]
      python tools/run_all_datasets.py                        # 运行全部
      python tools/run_all_datasets.py --algo ADFNR           # 仅运行指定算法
      python tools/run_all_datasets.py --dataset iris         # 仅运行指定数据集
      python tools/run_all_datasets.py --cpu                  # 强制使用 CPU
      python tools/run_all_datasets.py --dry-run              # 仅打印计划, 不实际运行
"""

import os
import sys
import time
import argparse
import importlib
import warnings
import gc
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ==================== 项目根目录 & GPU 检测 ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets"

# 检测 GPU
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "N/A"
except ImportError:
    torch = None
    CUDA_AVAILABLE = False
    GPU_NAME = "torch 未安装"

# ==================== 文件名映射 ====================
FILE_NAME_MAP = {
    "breast-cancer": "breast-cancer-wisconsin",
    "wine-red":      "winequality-red",
    "wine-white":    "winequality-white",
}

def resolve_filename(name):
    return FILE_NAME_MAP.get(name, name)

# ==================== 数据集配置 (原配置不变) ====================
DATASETS = [
    {"name": "adult",         "outlier_col": "income",   "outlier_val": ">50K"},
    {"name": "arrhythmia",    "outlier_col": "C280",     "outlier_val": "3,4,5,7,8,9,14,15"},
    {"name": "bank",          "outlier_col": "y",        "outlier_val": "yes"},
    {"name": "bank-full",     "outlier_col": "y",        "outlier_val": "yes"},
    {"name": "banknote",      "outlier_col": "class",    "outlier_val": "1"},
    {"name": "breast-cancer", "outlier_col": "Class",    "outlier_val": "4"},
    {"name": "car",           "outlier_col": "class",    "outlier_val": "good,vgood"},
    {"name": "chess",         "outlier_col": "won",      "outlier_val": "nowin"},
    {"name": "credit",        "outlier_col": "C16",      "outlier_val": "-"},
    {"name": "diabetes",      "outlier_col": "class",    "outlier_val": "Negative"},
    {"name": "german",        "outlier_col": "Class",    "outlier_val": "2"},
    {"name": "glass",         "outlier_col": "Type_of_glass", "outlier_val": "3,5,6"},
    {"name": "horse",         "outlier_col": "cp_data",  "outlier_val": "1"},
    {"name": "iris",          "outlier_col": "class",    "outlier_val": "Iris-setosa"},
    {"name": "mushroom",      "outlier_col": "class",    "outlier_val": "m,u,w"},
    {"name": "nursery",       "outlier_col": "class",    "outlier_val": "recommend,very_recom"},
    {"name": "parkinsons",    "outlier_col": "status",   "outlier_val": "0"},
    {"name": "raisin",        "outlier_col": "Class",    "outlier_val": "Besni"},
    {"name": "student-mat",   "outlier_col": "G3",       "outlier_val": "4,5,7,17,19,20"},
    {"name": "wine",          "outlier_col": "class",    "outlier_val": "3"},
    {"name": "wine-red",      "outlier_col": "quality",  "outlier_val": "3,4,8"},
    {"name": "wine-white",    "outlier_col": "quality",  "outlier_val": "3,4,8,9"},
    {"name": "yeast",         "outlier_col": "Class",    "outlier_val": "ERL"},
    {"name": "zoo",           "outlier_col": "type",     "outlier_val": "3,5,6"},
]

# ==================== 算法配置 ====================
ALGORITHMS = [
    {
        "name": "ADFNR",
        "module": "ADFNR.detector",
        "script_path": "ADFNR/detector.py",
        "uses_gpu": False,
        "init_kwargs": {},
    },
    {
        "name": "GCN",
        "module": "GCN.detector",
        "script_path": "GCN/detector.py",
        "uses_gpu": True,
        "init_kwargs": {},
    },
    {
        "name": "GCN-LOF",
        "module": "GCN-LOF.detector",
        "script_path": "GCN-LOF/detector.py",
        "uses_gpu": True,
        "init_kwargs": {},
    },
    {
        "name": "GCOD",
        "module": "GCOD.detector",
        "script_path": "GCOD/detector.py",
        "uses_gpu": False,
        "init_kwargs": {},
    },
    {
        "name": "IE",
        "module": "IE.detector",
        "script_path": "IE/detector.py",
        "uses_gpu": False,
        "init_kwargs": {},
    },
    {
        "name": "KNN",
        "module": "KNN.detector",
        "script_path": "KNN/detector.py",
        "uses_gpu": False,
        "init_kwargs": {},
    },
    {
        "name": "NIEOD",
        "module": "NIEOD.detector",
        "script_path": "NIEOD/detector.py",
        "uses_gpu": False,
        "init_kwargs": {},
    },
    {
        "name": "NMIGOD",
        "module": "NMIGOD.detector",
        "script_path": "NMIGOD/detector.py",
        "uses_gpu": True,
        "init_kwargs": {},
    },
    {
        "name": "DASOD",
        "module": "DASOD.detector",
        "script_path": "DASOD/detector.py",
        "uses_gpu": False,
        "init_kwargs": {"K": 5, "lambda_ratio": 0.05},
    },
]


def configure_framework(fw, ds_cfg, output_dir, force_cpu=False):
    """将数据集配置注入框架实例, 绕过交互式输入。"""
    name = ds_cfg["name"]
    actual_name = resolve_filename(name)
    csv_path = DATA_ROOT / f"{actual_name}.csv"

    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)
    anomaly_vals = [v.strip() for v in ds_cfg["outlier_val"].split(',')
                    if v.strip()]

    # 设置框架属性
    fw.df_raw = df
    fw.target_column = ds_cfg["outlier_col"]
    fw.anomaly_values = anomaly_vals
    fw.output_folder = str(output_dir)

    # 强制 CPU (如果指定)
    if force_cpu and hasattr(fw, 'device'):
        fw.device = torch.device('cpu')

    return True


def run_dataset_on_framework(fw, ds_cfg):
    """对单个数据集执行完整检测流水线。"""
    fw.preprocess_data()
    fw.train_model()
    fw.get_anomaly_scores()
    fw.optimize_threshold()
    y_pred = fw.calculate_metrics_and_topk()
    fw.save_results(y_pred)


def clear_gpu_memory():
    """清理 GPU 显存。"""
    if CUDA_AVAILABLE:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def import_algorithm_module(algo):
    """
    动态导入算法模块并返回 AnomalyDetectionFramework 类。
    使用 importlib 加载指定的 detector 模块。
    """
    script_path = PROJECT_ROOT / algo["script_path"]
    if not script_path.exists():
        raise FileNotFoundError(f"算法脚本不存在: {script_path}")

    # 构建模块名 (避免命名冲突)
    module_name = f"_algo_{algo['name'].replace('-', '_').lower()}"

    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.AnomalyDetectionFramework


def run_algorithm(algo, datasets, force_cpu=False, dry_run=False):
    """运行单个算法在所有数据集上。"""
    algo_name = algo["name"]
    print(f"\n{'='*60}")
    gpu_tag = ""
    if algo["uses_gpu"]:
        if force_cpu:
            gpu_tag = " [GPU可用但强制CPU]"
        elif CUDA_AVAILABLE:
            gpu_tag = f" [GPU: {GPU_NAME}]"
        else:
            gpu_tag = " [CPU — CUDA不可用]"
    print(f"  算法: {algo_name}{gpu_tag}")
    print(f"  脚本: {algo['script_path']}")
    print(f"{'='*60}")

    if dry_run:
        for ds in datasets:
            name = ds["name"]
            actual_name = resolve_filename(name)
            csv_path = DATA_ROOT / f"{actual_name}.csv"
            status = "OK" if csv_path.exists() else "MISSING"
            out_dir = PROJECT_ROOT / algo_name / f"output_{name}"
            print(f"  [DRY-RUN] {ds['name']:<20s} -> {out_dir}  ({status})")
        return {"success": len(datasets), "fail": 0, "skip": 0}

    # ----- 导入算法模块 -----
    t0 = time.time()
    print(f"  导入模块...")
    try:
        FrameworkClass = import_algorithm_module(algo)
    except Exception as e:
        print(f"  [ERROR] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": 0, "fail": len(datasets), "skip": 0}
    print(f"  模块导入耗时: {time.time() - t0:.1f}s")

    success, fail, skip = 0, 0, 0
    algo_start = time.time()

    for idx, ds_cfg in enumerate(datasets, 1):
        ds_name = ds_cfg["name"]
        out_dir = PROJECT_ROOT / algo_name / f"output_{ds_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{idx}/{len(datasets)}] {ds_name:<22s}", end=" ", flush=True)

        # 创建框架实例并配置
        try:
            fw = FrameworkClass(**algo.get("init_kwargs", {}))
        except TypeError:
            fw = FrameworkClass()  # 如果不接受 kwargs

        if not configure_framework(fw, ds_cfg, out_dir, force_cpu=force_cpu):
            print("[SKIP] 文件不存在")
            skip += 1
            continue

        # 执行流水线
        try:
            t_start = time.time()
            run_dataset_on_framework(fw, ds_cfg)
            elapsed = time.time() - t_start
            print(f"[OK] {elapsed:.1f}s")
            success += 1
        except Exception as e:
            print(f"[FAIL] {str(e)[:100]}")
            fail += 1

        # 清理当前框架实例
        del fw
        gc.collect()

    algo_elapsed = time.time() - algo_start
    print(f"  算法 {algo_name} 完成 | "
          f"成功: {success} | 失败: {fail} | 跳过: {skip} | "
          f"耗时: {algo_elapsed/60:.1f}分钟")

    # GPU 算法运行完毕后清理显存
    if algo["uses_gpu"] and not force_cpu:
        clear_gpu_memory()

    # 从 sys.modules 中移除模块, 避免下次导入冲突
    module_name = f"_algo_{algo['name'].replace('-', '_').lower()}"
    sys.modules.pop(module_name, None)
    gc.collect()

    return {"success": success, "fail": fail, "skip": skip}


def main():
    parser = argparse.ArgumentParser(
        description="批量异常检测 — 逐算法导入执行 (GPU加速)")
    parser.add_argument("--algo", type=str, default=None,
                        help="仅运行指定算法 (如 ADFNR, GCN, ...)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="仅运行指定数据集 (如 iris, adult, ...)")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU (禁用 GPU 加速)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印计划, 不实际运行")
    args = parser.parse_args()

    # 过滤算法
    if args.algo:
        algos = [a for a in ALGORITHMS if a["name"].upper() == args.algo.upper()]
        if not algos:
            print(f"错误: 未找到算法 '{args.algo}'。"
                  f"可用: {', '.join(a['name'] for a in ALGORITHMS)}")
            return
    else:
        algos = ALGORITHMS

    # 过滤数据集
    if args.dataset:
        dss = [d for d in DATASETS if d["name"] == args.dataset]
        if not dss:
            names = [d["name"] for d in DATASETS]
            print(f"错误: 未找到数据集 '{args.dataset}'。"
                  f"可用: {', '.join(names)}")
            return
    else:
        dss = DATASETS

    total = len(algos) * len(dss)

    # 启动信息
    print(f"{'='*60}")
    print(f"  批量异常检测任务 (逐算法导入模式)")
    print(f"  算法: {len(algos)} 个 | 数据集: {len(dss)} 个 | 总任务: {total} 个")
    if CUDA_AVAILABLE and not args.cpu:
        print(f"  GPU 加速: 已启用 ({GPU_NAME})")
    else:
        print(f"  GPU 加速: {'已禁用 (--cpu)' if args.cpu else '不可用'}")
    if args.dry_run:
        print(f"  *** DRY-RUN 模式: 仅打印计划 ***")
    print(f"{'='*60}")

    overall_start = time.time()
    total_success, total_fail, total_skip = 0, 0, 0

    for algo in algos:
        result = run_algorithm(algo, dss, force_cpu=args.cpu,
                               dry_run=args.dry_run)
        total_success += result["success"]
        total_fail += result["fail"]
        total_skip += result["skip"]

    overall_elapsed = time.time() - overall_start

    print(f"\n{'='*60}")
    print(f"  全部完成!")
    print(f"  成功: {total_success} | 失败: {total_fail} | "
          f"跳过: {total_skip} | 总计: {total}")
    print(f"  总耗时: {overall_elapsed/60:.1f} 分钟 ({overall_elapsed:.0f} 秒)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
