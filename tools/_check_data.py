#!/usr/bin/env python3
"""Temporary script to verify CSV data integrity across algorithms and datasets."""
import pandas as pd
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent
algos_with_data = ['ADFNR', 'DASOD', 'GCN', 'GCN-LOF', 'NIEOD', 'NMIGOD']

datasets_all = sorted([d.name.replace('output_','') for d in (root/'ADFNR').glob('output_*')])
print(f'Total datasets: {len(datasets_all)}')
print()

issues = []
for ds in datasets_all:
    rows_set = set()
    anomaly_counts = set()
    for algo in algos_with_data:
        csv_path = root / algo / f'output_{ds}' / 'detection_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            rows_set.add(len(df))
            actuals = df.iloc[:, 3].astype(float).values
            anomaly_counts.add(int((actuals > 0).sum()))
    if len(rows_set) > 1:
        print(f'ROW COUNT MISMATCH: {ds}: {rows_set}')
        issues.append((ds, 'row_count'))
    if len(anomaly_counts) > 1:
        print(f'ANOMALY COUNT MISMATCH: {ds}: {anomaly_counts}')
        issues.append((ds, 'anomaly_count'))
    n = list(rows_set)[0] if rows_set else 0
    a = list(anomaly_counts)[0] if anomaly_counts else 0
    pct = a/n*100 if n else 0
    status = 'OK'
    if a == 0:
        status = 'ZERO_ANOMALIES'
        issues.append((ds, 'zero_anomalies'))
    elif pct < 1:
        status = f'LOW_ANOMALY({pct:.2f}%)'
    print(f'  {ds:20s}: {n:5d} rows, {a:5d} anomalies ({pct:5.1f}%) - {status}')

print(f'\nTotal issues found: {len(issues)}')
for ds, issue in issues:
    print(f'  - {ds}: {issue}')

# Also check that each algorithm has all 24 datasets
print(f'\n--- Per-algo coverage ---')
for algo in algos_with_data:
    count = len(list((root/algo).glob('output_*/detection_results.csv')))
    missing = set(datasets_all) - set(d.name.replace('output_','') for d in (root/algo).glob('output_*'))
    print(f'  {algo}: {count}/24 datasets')
    if missing:
        print(f'    Missing: {missing}')

# Check CSV column counts
print(f'\n--- CSV column count check ---')
for algo in algos_with_data[:1]:
    for ds in datasets_all[:1]:
        csv_path = root / algo / f'output_{ds}' / 'detection_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f'  {algo}/{ds}: {df.shape[1]} columns: {list(df.columns)}')
