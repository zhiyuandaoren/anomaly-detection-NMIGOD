#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMIGOD 消融实验 + 交叉验证
============================================================
三个变体在 30 个数据集上对比:
  1. NMIGOD-full:  自适应半径 + NMI图 + GCN
  2. NMIGOD-noAda: 固定半径 ε=σ_a (无自适应)
  3. NMIGOD-noGCN: 纯 NMI 分数 (无 GCN)

用法:
  python tools/nmigod_ablation.py
"""

import os, sys, time, gc, shutil, json, warnings
import importlib
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets"

try:
    import torch
    CUDA_OK = torch.cuda.is_available()
except ImportError:
    CUDA_OK = False

FILE_NAME_MAP = {
    "breast-cancer": "breast-cancer-wisconsin",
    "wine-red": "winequality-red",
    "wine-white": "winequality-white",
}

def resolve_filename(name):
    return FILE_NAME_MAP.get(name, name)

# 全量数据集
ALL_DATASETS = [
    {"name": d, "outlier_col": "class", "outlier_val": "anomaly"}
    for d in [
        "abalone", "adult", "arrhythmia", "bank", "bank-full", "banknote",
        "breast-cancer", "car", "chess", "cmc", "covertype", "credit",
        "diabetes", "german", "glass", "heart", "hepatitis", "horse",
        "iris", "mushroom", "nursery", "parkinsons", "raisin", "skin",
        "student-mat", "wine", "wine-red", "wine-white", "yeast", "zoo",
    ]
]


def run_variant(variant_name, extra_kwargs, best_params):
    """运行一个消融变体在所有 30 个数据集上。"""
    print(f"\n{'='*60}")
    print(f"  消融变体: {variant_name}")
    print(f"  kwargs: {extra_kwargs}")
    print(f"{'='*60}")

    script_path = PROJECT_ROOT / "NMIGOD" / "detector.py"
    module_name = f"_abl_{variant_name.replace('-','_').lower()}"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    FrameworkClass = module.AnomalyDetectionFramework

    results = []
    for idx, ds_cfg in enumerate(ALL_DATASETS):
        ds_name = ds_cfg["name"]
        out_dir = PROJECT_ROOT / "NMIGOD" / f"output_abl_{variant_name}_{ds_name}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        actual_name = resolve_filename(ds_name)
        csv_path = DATA_ROOT / f"{actual_name}.csv"
        if not csv_path.exists():
            continue

        kw = dict(best_params) if best_params else {}
        kw.update(extra_kwargs)
        fw = FrameworkClass(**kw)

        df = pd.read_csv(csv_path)
        fw.df_raw = df
        fw.target_column = ds_cfg["outlier_col"]
        fw.anomaly_values = [v.strip() for v in ds_cfg["outlier_val"].split(',') if v.strip()]
        fw.output_folder = str(out_dir)

        try:
            t1 = time.time()
            fw.preprocess_data()
            fw.train_model()
            fw.get_anomaly_scores()
            fw.optimize_threshold()
            fw.calculate_metrics_and_topk()
            elapsed = time.time() - t1

            mpath = out_dir / "metrics.csv"
            m = pd.read_csv(mpath) if mpath.exists() else pd.DataFrame()
            if len(m):
                m_dict = dict(zip(m["Metric"], m["Value"]))
            else:
                m_dict = {"F1-Score": 0, "AUC": 0, "Precision": 0, "Recall": 0}

            f1 = m_dict.get("F1-Score", 0)
            auc = m_dict.get("AUC", 0)
            print(f"  [{idx+1}/30] {ds_name:<20s} "
                  f"F1={f1:.4f} AUC={auc:.4f} ({elapsed:.0f}s)")
            results.append({
                "dataset": ds_name, "variant": variant_name,
                "F1": round(f1, 4), "AUC": round(auc, 4),
                "Precision": round(m_dict.get("Precision", 0), 4),
                "Recall": round(m_dict.get("Recall", 0), 4),
            })
        except Exception as e:
            print(f"  [{idx+1}/30] {ds_name}: ERROR {str(e)[:80]}")

        del fw; gc.collect()
        if CUDA_OK:
            torch.cuda.empty_cache()

    avg_f1 = np.mean([r["F1"] for r in results]) if results else 0
    avg_auc = np.mean([r["AUC"] for r in results]) if results else 0
    print(f"  [{variant_name}] avg F1={avg_f1:.4f} avg AUC={avg_auc:.4f}")
    return results


def cross_validation(k_folds=5, datasets=None):
    """5-fold 分层交叉验证 (仅 NMIGOD 完整版, 8个代表数据集)。"""
    from sklearn.model_selection import StratifiedKFold

    if datasets is None:
        datasets = ["iris", "wine", "glass", "diabetes",
                    "german", "mushroom", "covertype", "adult"]

    print(f"\n{'='*60}")
    print(f"  {k_folds}-Fold 交叉验证 (NMIGOD)")
    print(f"{'='*60}")

    cv_results = []
    for ds_name in datasets:
        actual_name = resolve_filename(ds_name)
        csv_path = DATA_ROOT / f"{actual_name}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        y = (df["class"] == "anomaly").astype(int).values
        X_idx = np.arange(len(df))

        skf = StratifiedKFold(n_splits=min(k_folds, y.sum(), (y == 0).sum()),
                             shuffle=True, random_state=42)
        fold_scores = []
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X_idx, y)):
            fw = FrameworkClass(random_state=42)
            fw.df_raw = df.iloc[train_val_idx].reset_index(drop=True)
            fw.target_column = "class"
            fw.anomaly_values = ["anomaly"]
            out_dir = PROJECT_ROOT / "NMIGOD" / f"output_cv_{ds_name}_fold{fold}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            fw.output_folder = str(out_dir)

            try:
                fw.preprocess_data()
                fw.train_model()
                fw.get_anomaly_scores()
                fw.optimize_threshold()
                fw.calculate_metrics_and_topk()
                mpath = out_dir / "metrics.csv"
                m = pd.read_csv(mpath) if mpath.exists() else pd.DataFrame()
                f1 = dict(zip(m["Metric"], m["Value"])).get("F1-Score", 0) if len(m) else 0
                fold_scores.append(f1)
            except Exception as e:
                print(f"    fold {fold}: ERROR {e}")

            del fw; gc.collect()
            if CUDA_OK:
                torch.cuda.empty_cache()

        if fold_scores:
            mu, std = np.mean(fold_scores), np.std(fold_scores)
            print(f"  {ds_name:<20s} F1={mu:.4f} ± {std:.4f} ({len(fold_scores)} folds)")
            cv_results.append({
                "dataset": ds_name, "cv_mean_f1": round(mu, 4),
                "cv_std_f1": round(std, 4), "n_folds": len(fold_scores)
            })

    out_path = PROJECT_ROOT / "metrics" / "cv_results.csv"
    pd.DataFrame(cv_results).to_csv(out_path, index=False)
    print(f"\nCV 结果已保存: {out_path}")
    return cv_results


def main():
    print("NMIGOD 消融实验 + 交叉验证")
    print(f"  GPU: {'可用' if CUDA_OK else '不可用'}")

    # 加载最佳参数
    bp_path = PROJECT_ROOT / "metrics" / "best_params.csv"
    best_params = None
    if bp_path.exists():
        bp_df = pd.read_csv(bp_path)
        nm_row = bp_df[bp_df["algorithm"] == "NMIGOD"]
        if len(nm_row) > 0:
            best_params = json.loads(nm_row.iloc[0]["params"])
            print(f"  最佳参数: {best_params}")
    if best_params is None:
        best_params = {"lambda_param": 1.0, "mi_threshold": 0.05,
                       "hidden1": 128, "hidden2": 64, "epochs": 200, "lr": 0.01}
        print(f"  默认参数: {best_params}")

    all_results = []
    t0 = time.time()

    # 三个变体
    variants = [
        ("NMIGOD-full",  {"use_adaptive_radius": True,  "use_gcn": True}),
        ("NMIGOD-noAda", {"use_adaptive_radius": False, "use_gcn": True}),
        ("NMIGOD-noGCN", {"use_adaptive_radius": True,  "use_gcn": False}),
    ]

    for vname, vkwargs in variants:
        results = run_variant(vname, vkwargs, best_params)
        all_results.extend(results)

    elapsed = time.time() - t0
    print(f"\n消融实验总耗时: {elapsed/60:.1f} 分钟")

    # 保存
    df = pd.DataFrame(all_results)
    out_path = PROJECT_ROOT / "metrics" / "ablation_results.csv"
    df.to_csv(out_path, index=False)

    # 汇总
    print(f"\n{'='*60}")
    print("消融实验汇总 (30 数据集平均):")
    for vname, _ in variants:
        sub = df[df["variant"] == vname]
        print(f"  {vname:<20s} F1={sub['F1'].mean():.4f}  "
              f"AUC={sub['AUC'].mean():.4f}  "
              f"Prec={sub['Precision'].mean():.4f}  "
              f"Rec={sub['Recall'].mean():.4f}")

    # 交叉验证
    cv_results = cross_validation(k_folds=5)

    return df, cv_results


if __name__ == "__main__":
    main()
