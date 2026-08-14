#!/usr/bin/env python3
"""Run GCN, GCN-LOF, NMIGOD on 5 large-scale datasets (50k-100k samples)."""
import os, sys, time, gc, importlib
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets_large"
OUTPUT_ROOT = PROJECT_ROOT / "large_results"

# Check GPU
try:
    import torch
    CUDA = torch.cuda.is_available()
    GPU = torch.cuda.get_device_name(0) if CUDA else "N/A"
except:
    torch = None; CUDA = False; GPU = "torch not installed"

# Datasets
DATASETS = [
    {"name": "shuttle",       "csv": "shuttle.csv",       "target": "class", "anomaly_val": "anomaly"},
    {"name": "connect4",      "csv": "connect4.csv",      "target": "class", "anomaly_val": "anomaly"},
    {"name": "covertype_80k", "csv": "covertype_80k.csv", "target": "class", "anomaly_val": "anomaly"},
    {"name": "electricity",   "csv": "electricity.csv",   "target": "class", "anomaly_val": "anomaly"},
    {"name": "sensor_70k",    "csv": "sensor_70k.csv",    "target": "class", "anomaly_val": "anomaly"},
]

# Algorithms (GCN-based only)
ALGOS = [
    {"name": "GCN",     "module": "GCN.detector",     "script": "GCN/detector.py",
     "gpu": True,  "kwargs": {"k_neighbors": 10, "hidden1": 128, "hidden2": 64,
                               "epochs": 200, "lr": 0.01}},
    {"name": "GCN-LOF", "module": "GCN-LOF.detector", "script": "GCN-LOF/detector.py",
     "gpu": True,  "kwargs": {"k_neighbors": 20, "lof_neighbors": 30,
                               "hidden1": 128, "hidden2": 64, "epochs": 200, "lr": 0.001}},
    {"name": "NMIGOD",  "module": "NMIGOD.detector",  "script": "NMIGOD/detector.py",
     "gpu": True,  "kwargs": {"lambda_param": 1.0, "mi_threshold": 0.03,
                               "hidden1": 128, "hidden2": 64, "epochs": 200, "lr": 0.001}},
]


def import_algo(algo):
    script_path = PROJECT_ROOT / algo["script"]
    module_name = f"_large_{algo['name'].replace('-','_').lower()}"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.AnomalyDetectionFramework


def run_one(algo, ds_cfg):
    """Run one algorithm on one dataset."""
    algo_name = algo["name"]
    ds_name = ds_cfg["name"]
    out_dir = OUTPUT_ROOT / algo_name / f"output_{ds_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / ds_cfg["csv"]
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return False

    df = pd.read_csv(csv_path)
    anomaly_vals = [ds_cfg["anomaly_val"]]

    try:
        FrameworkClass = import_algo(algo)
        fw = FrameworkClass(**algo.get("kwargs", {}))
    except TypeError:
        fw = FrameworkClass()

    fw.df_raw = df
    fw.target_column = ds_cfg["target"]
    fw.anomaly_values = anomaly_vals
    fw.output_folder = str(out_dir)

    try:
        t0 = time.time()
        fw.preprocess_data()
        fw.train_model()
        fw.get_anomaly_scores()
        fw.optimize_threshold()
        fw.calculate_metrics_and_topk()
        fw.save_results((fw.scores >= fw.best_threshold).astype(int))
        elapsed = time.time() - t0
        print(f"  [{algo_name}] {ds_name}: OK ({elapsed:.0f}s)")
        # Read back metrics for display
        m = pd.read_csv(out_dir / "metrics.csv")
        vals = {r["Metric"]: r["Value"] for _, r in m.iterrows()}
        print(f"    P={vals.get('Precision',0):.4f} R={vals.get('Recall',0):.4f} "
              f"F1={vals.get('F1-Score',0):.4f} AUC={vals.get('AUC',0):.4f}")
        return True
    except Exception as e:
        print(f"  [{algo_name}] {ds_name}: FAIL - {str(e)[:120]}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        del fw; gc.collect()
        if CUDA: torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("  Large-Scale Experiment: GCN | GCN-LOF | NMIGOD")
    print(f"  Datasets: {len(DATASETS)} ({', '.join(d['name'] for d in DATASETS)})")
    print(f"  GPU: {GPU}")
    print("=" * 60)

    total = len(ALGOS) * len(DATASETS)
    overall_start = time.time()
    success, fail = 0, 0

    for algo in ALGOS:
        print(f"\n--- {algo['name']} ---")
        for ds in DATASETS:
            ok = run_one(algo, ds)
            if ok:
                success += 1
            else:
                fail += 1

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  Complete! Success: {success}/{total} | Fail: {fail}/{total}")
    print(f"  Time: {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"  Results: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
