#!/usr/bin/env python3
"""
Reformat Top-K tables to wide format with averages.

Columns: Dataset, Top_K%, ADFNR_P, ADFNR_R, ADFNR_F1, DASOD_P, ..., NMIGOD_F1
Each dataset has an "Average" row (mean across all K values).
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / 'metrics'

# ── 24 datasets ──
DATASETS_24 = [
    "adult", "bank-full", "banknote", "breast-cancer", "car", "chess",
    "cmc", "credit", "diabetes", "german", "glass", "heart",
    "horse", "iris", "mushroom", "nursery", "parkinsons", "raisin",
    "student-mat", "wine", "wine-red", "wine-white", "yeast", "zoo",
]

MIXED  = {"adult", "bank-full", "car", "cmc", "credit", "diabetes",
          "german", "heart", "student-mat", "yeast"}
NUMER  = {"banknote", "breast-cancer", "glass", "horse", "iris",
          "parkinsons", "raisin", "wine", "wine-red", "wine-white"}
CATGR  = {"chess", "mushroom", "nursery", "zoo"}

def get_dtype(ds):
    if ds in MIXED: return "Mixed"
    elif ds in NUMER: return "Numerical"
    else: return "Categorical"

ALGOS = ['ADFNR', 'DASOD', 'GCN', 'GCN-LOF', 'NIEOD', 'NMIGOD']
METRICS = ['P', 'R', 'F1']
STANDARD_PCTS = [1, 3, 5, 10, 20, 30, 50]


def collect_all():
    rows = []
    for ds in DATASETS_24:
        dtype = get_dtype(ds)
        for algo in ALGOS:
            p = PROJECT_ROOT / algo / f'output_{ds}' / 'topk_metrics.csv'
            if not p.exists():
                continue
            df = pd.read_csv(p)
            pct_col = 'Percentage(%)' if 'Percentage(%)' in df.columns else 'Percentage'
            for _, row in df.iterrows():
                pct = row[pct_col]
                dists = [abs(pct - sp) for sp in STANDARD_PCTS]
                best_i = np.argmin(dists)
                if dists[best_i] < 1.5:
                    rows.append({
                        'Dataset': ds, 'Type': dtype, 'Algorithm': algo,
                        'Top_K%': STANDARD_PCTS[best_i],
                        'P': round(row['Precision'], 4),
                        'R': round(row['Recall'], 4),
                        'F1': round(row['F1-Score'], 4),
                    })
    return rows


def build_wide(df_long):
    """Convert long format to wide: one row per (Dataset, Top_K%)."""
    # Pivot: each (Dataset, Top_K%) gets columns for each (Algorithm, Metric)
    wide_rows = []
    for (ds, k), grp in df_long.groupby(['Dataset', 'Top_K%']):
        row = {'Dataset': ds, 'Top_K%': int(k)}
        for _, r in grp.iterrows():
            algo = r['Algorithm']
            for m in METRICS:
                row[f'{algo}_{m}'] = r[m]
        wide_rows.append(row)
    return wide_rows


def add_averages(df_long, wide_rows):
    """For each dataset, compute average across K for each algorithm, add as Top_K%='Average'."""
    avg_rows = []
    for ds, grp in df_long.groupby('Dataset'):
        row = {'Dataset': ds, 'Top_K%': 'Average'}
        for algo in ALGOS:
            algo_data = grp[grp['Algorithm'] == algo]
            if len(algo_data) > 0:
                for m in METRICS:
                    row[f'{algo}_{m}'] = round(algo_data[m].mean(), 4)
            else:
                for m in METRICS:
                    row[f'{algo}_{m}'] = '-'
        avg_rows.append(row)
    return avg_rows


def main():
    print("Collecting Top-K data...")
    rows = collect_all()
    df_all = pd.DataFrame(rows)
    df_all = df_all.drop_duplicates(subset=['Dataset', 'Algorithm', 'Top_K%'], keep='first')
    print(f"  Collected {len(df_all)} rows")

    # Column order
    col_order = ['Dataset', 'Top_K%']
    for algo in ALGOS:
        for m in METRICS:
            col_order.append(f'{algo}_{m}')

    for dtype in ['Mixed', 'Numerical', 'Categorical']:
        df_type = df_all[df_all['Type'] == dtype].copy()
        n_ds = df_type['Dataset'].nunique()

        # Build wide rows
        wide = build_wide(df_type)
        # Add average rows
        avg = add_averages(df_type, wide)
        # Combine: K rows first, then Average per dataset
        all_rows = wide + avg

        df_out = pd.DataFrame(all_rows, columns=col_order)
        # Sort: by dataset, then K values (with Average last per dataset)
        # Create sort key
        ds_order = {ds: i for i, ds in enumerate(DATASETS_24)}
        k_order = {**{p: p for p in STANDARD_PCTS}, 'Average': 999}

        df_out['_ds_sort'] = df_out['Dataset'].map(ds_order)
        df_out['_k_sort'] = df_out['Top_K%'].map(k_order)
        df_out = df_out.sort_values(['_ds_sort', '_k_sort'])
        df_out = df_out.drop(columns=['_ds_sort', '_k_sort'])

        # Fill NaN with '-'
        df_out = df_out.fillna('-')

        fname = f'topk_{dtype.lower()}.csv'
        fpath = OUT_DIR / fname
        df_out.to_csv(fpath, index=False)
        print(f"  [{dtype}] {len(df_out)} rows ({n_ds} datasets × {len(STANDARD_PCTS)+1} K-levels) -> {fpath}")

    print("\n[DONE]")


if __name__ == '__main__':
    main()
