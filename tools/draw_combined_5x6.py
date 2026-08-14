#!/usr/bin/env python3
"""
Regenerate combined figures (4×7 = 28 datasets, minus covertype & skin)
+ Generate 3 Top-K tables grouped by data type (Mixed / Numerical / Categorical).

Changes:
  - 28 datasets only (removed covertype, skin)
  - Layout: 4 columns × 7 rows
  - Larger fonts, legend placed at bottom
  - 3 Top‑K tables by data type, saved to large_results/ (or metrics/)
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_ROOT = PROJECT_ROOT / 'images'
METRICS_DIR = PROJECT_ROOT / 'metrics'

# ── 28 datasets (exclude covertype, skin) ──────────────────────
DATASETS_28 = [
    "abalone", "adult", "arrhythmia", "bank", "bank-full", "banknote",
    "breast-cancer", "car", "chess", "cmc", "credit",
    "diabetes", "german", "glass", "heart", "hepatitis", "horse",
    "iris", "mushroom", "nursery", "parkinsons", "raisin",
    "student-mat", "wine", "wine-red", "wine-white", "yeast", "zoo",
]  # sorted alphabetically

# ── Data-type grouping (from EXPERIMENTS_COMPLETE.md Table 5) ──
DATA_TYPE = {}
_mixed  = {"adult","arrhythmia","bank","bank-full","car","credit",
           "diabetes","german","student-mat","yeast","abalone","heart",
           "cmc","hepatitis"}
_numer  = {"breast-cancer","banknote","glass","horse","iris","parkinsons",
           "raisin","wine","wine-red","wine-white"}
_catgr  = {"chess","mushroom","nursery","zoo"}
for d in DATASETS_28:
    if d in _mixed:   DATA_TYPE[d] = "Mixed"
    elif d in _numer: DATA_TYPE[d] = "Numerical"
    else:             DATA_TYPE[d] = "Categorical"

# ── Algorithms & colours ───────────────────────────────────────
ALGOS = ['ADFNR','DASOD','GCN','GCN-LOF','NIEOD','NMIGOD']
ALGO_COLORS = {
    'ADFNR':'#1f77b4','DASOD':'#ff7f0e','GCN':'#2ca02c',
    'GCN-LOF':'#9467bd','NIEOD':'#17becf','NMIGOD':'#E31818',
}
ALGO_LW    = {a: (3.5 if a=='NMIGOD' else 1.8) for a in ALGOS}
ALGO_ALPHA = {a: (0.95 if a=='NMIGOD' else 0.65) for a in ALGOS}
ALGO_Z     = {a: (10   if a=='NMIGOD' else 5)    for a in ALGOS}

# ── Metric config ─────────────────────────────────────────────
METRIC_CFG = {
    'precision': {'key':'prec','title':'Precision','ylabel':'Precision',
                  'fname':'combined_4x7_precision'},
    'recall':    {'key':'rec', 'title':'Recall',   'ylabel':'Recall',
                  'fname':'combined_4x7_recall'},
    'f1':        {'key':'f1',  'title':'F1-Score', 'ylabel':'F1-Score',
                  'fname':'combined_4x7_f1'},
    'roc':       {'key':'roc', 'title':'ROC Curve', 'ylabel':'TPR',
                  'fname':'combined_4x7_roc'},
}

# =====================================================================
def load_one(csv_path):
    df = pd.read_csv(csv_path)
    raw_scores = df.iloc[:,1].astype(float).values
    raw_actuals = df.iloc[:,3].astype(float).values
    idx = np.argsort(-raw_scores)
    scores = raw_scores[idx]
    actuals = (raw_actuals[idx] > 0).astype(int)
    return scores, actuals

def compute_prf(scores, actuals):
    n = len(scores)
    total = int(actuals.sum())
    if total == 0: return np.zeros(n),np.zeros(n),np.zeros(n)
    tp = np.cumsum(actuals)
    k = np.arange(1,n+1,dtype=np.float64)
    prec = tp/k
    rec  = tp/total
    f1   = 2*prec*rec/np.maximum(prec+rec,1e-15)
    return prec, rec, f1

def downsample(arr, max_pts=200):
    n=len(arr)
    if n<=max_pts: return np.arange(n),arr
    step=max(1,n//max_pts)
    idx=np.arange(0,n,step)
    if idx[-1]!=n-1: idx=np.append(idx,n-1)
    return idx,arr[idx]

def load_all():
    all_data={}
    for ds in DATASETS_28:
        ds_data={}
        for algo in ALGOS:
            p = PROJECT_ROOT/algo/f'output_{ds}'/'detection_results.csv'
            if not p.exists(): continue
            sc,ac = load_one(str(p))
            if sc is None: continue
            prec,rec,f1 = compute_prf(sc,ac)
            fpr,tpr,_   = roc_curve(ac,sc)
            roc_auc     = auc(fpr,tpr)
            ds_data[algo]={'prec':prec,'rec':rec,'f1':f1,
                           'fpr':fpr,'tpr':tpr,'auc':roc_auc,'n':len(sc)}
        if ds_data: all_data[ds]=ds_data
    return all_data

# =====================================================================
# 1. TOP‑K TABLES  (by data type)
# =====================================================================
def gen_topk_tables(out_dir, fmt):
    """Read NMIGOD topk_metrics.csv for each dataset, group by data type, save 3 CSVs."""
    import csv
    topk_rows = {'Mixed':[], 'Numerical':[], 'Categorical':[]}

    for ds in DATASETS_28:
        p = PROJECT_ROOT/'NMIGOD'/f'output_{ds}'/'topk_metrics.csv'
        if not p.exists(): continue
        df = pd.read_csv(p)
        tp = DATA_TYPE.get(ds,'Mixed')
        # Keep standard K percentages: 1,3,5,10,20,30,50 (closest match ±1.5%)
        standard_pcts = [1, 3, 5, 10, 20, 30, 50]
        best_for_ds = {}  # (ds, matched_pct) -> (dist, row_data)
        for _,row in df.iterrows():
            pct = row.get('Percentage(%)',row.get('Percentage',0))
            dists = [abs(pct - sp) for sp in standard_pcts]
            best_i = np.argmin(dists)
            if dists[best_i] < 1.5:
                matched = standard_pcts[best_i]
                key = (ds, matched)
                row_data = {
                    'Dataset':ds,'Top_K%':matched,
                    'Precision':row['Precision'],'Recall':row['Recall'],
                    'F1-Score':row['F1-Score'],'AUC':row.get('AUC',0),
                }
                if key not in best_for_ds or dists[best_i] < best_for_ds[key][0]:
                    best_for_ds[key] = (dists[best_i], row_data)
        for (ds,matched), (dist, row_data) in best_for_ds.items():
            topk_rows[tp].append(row_data)

    os.makedirs(out_dir, exist_ok=True)
    for tp in ['Mixed','Numerical','Categorical']:
        rows = topk_rows[tp]
        if not rows: continue
        df = pd.DataFrame(rows)
        # Sort by dataset then Top_K%
        df = df.sort_values(['Dataset','Top_K%'])
        fpath = os.path.join(out_dir, f'topk_{tp.lower()}.{fmt}')
        if fmt=='csv':
            df.to_csv(fpath, index=False, float_format='%.4f')
        else:
            df.to_csv(fpath, index=False, float_format='%.4f')
        print(f"  Top-K [{tp}] -> {fpath}  ({len(df)} rows)")

# =====================================================================
# 2. COMBINED 4×7 FIGURES  (Precision / Recall / F1 / ROC)
# =====================================================================
def draw_combined(all_data, out_dir, fmt):
    n_rows, n_cols = 7, 4  # 7 rows × 4 cols = 28

    for metric_name in ['precision','recall','f1','roc']:
        cfg = METRIC_CFG[metric_name]
        key = cfg['key']; title = cfg['title']; ylabel = cfg['ylabel']

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 48))
        axes = axes.flatten()

        for idx, ds_name in enumerate(DATASETS_28):
            ax = axes[idx]
            if ds_name not in all_data:
                ax.text(0.5,0.5,'No Data',transform=ax.transAxes,
                        ha='center',va='center',fontsize=16,color='gray')
                ax.set_xlim(0,1); ax.set_ylim(0,1)
                ax.set_xlabel(ds_name, fontsize=13, fontweight='bold')
                continue

            ds_data = all_data[ds_name]
            algos_present = [a for a in ALGOS if a in ds_data]

            if metric_name == 'roc':
                for algo in algos_present:
                    d = ds_data[algo]
                    idx_p,fpr_p = downsample(d['fpr'])
                    tpr_p = d['tpr'][idx_p]
                    ax.plot(fpr_p, tpr_p,
                            label=f"{algo} ({d['auc']:.3f})",
                            color=ALGO_COLORS[algo],
                            linewidth=ALGO_LW[algo],
                            alpha=ALGO_ALPHA[algo],
                            zorder=ALGO_Z[algo])
                ax.plot([0,1],[0,1],'k:',linewidth=0.6,alpha=0.3)
            else:
                for algo in algos_present:
                    d = ds_data[algo]
                    y_vals = d[key]
                    k_norm = np.linspace(0,1,len(y_vals))
                    idx_p,y_p = downsample(y_vals)
                    ax.plot(k_norm[idx_p], y_p, label=algo,
                            color=ALGO_COLORS[algo],
                            linewidth=ALGO_LW[algo],
                            alpha=ALGO_ALPHA[algo],
                            zorder=ALGO_Z[algo])
                ax.set_ylim(0,1.05)
                ax.set_yticks([0,0.5,1])

            ax.set_xlim(0,1)
            ax.set_xticks([0,0.5,1])
            ax.tick_params(labelsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlabel(ds_name, fontsize=13, fontweight='bold', labelpad=4)

        # Hide unused
        for idx in range(len(DATASETS_28), n_rows*n_cols):
            axes[idx].set_visible(False)

        # ── Suptitle ──
        fig.suptitle(f'{title} — 28 Datasets (6 Algorithms)',
                     fontsize=32, fontweight='bold', y=0.998)

        # ── Global axis labels ──
        if metric_name == 'roc':
            fig.text(0.5, 0.008, 'False Positive Rate', ha='center',
                     fontsize=20, fontweight='bold')
            fig.text(0.008, 0.5, 'True Positive Rate', va='center',
                     rotation='vertical', fontsize=20, fontweight='bold')
        else:
            fig.text(0.5, 0.008, 'Normalized Threshold Index (k / N)',
                     ha='center', fontsize=20, fontweight='bold')
            fig.text(0.008, 0.5, ylabel, va='center',
                     rotation='vertical', fontsize=20, fontweight='bold')

        # ── Common legend at bottom ──
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center',
                       ncol=6, fontsize=18, framealpha=0.9,
                       bbox_to_anchor=(0.5, -0.005), borderpad=0.5)

        plt.subplots_adjust(left=0.04, right=0.98, top=0.97, bottom=0.04,
                            wspace=0.28, hspace=0.42)

        out_path = os.path.join(out_dir, f'{cfg["fname"]}.{fmt}')
        fig.savefig(out_path, dpi=180, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  [{metric_name}] -> {out_path}")

# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir','-o', default=str(IMAGES_ROOT))
    parser.add_argument('--fmt','-f', default='svg')
    parser.add_argument('--step', choices=['tables','figures','all'], default='all')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  28 datasets (no covertype, no skin)")
    print(f"  Layout: 7 rows × 4 columns")
    print(f"  Data types: Mixed=14, Numerical=10, Categorical=4")
    print(f"{'='*60}")

    # ---- Generate Top-K tables ----
    if args.step in ('tables','all'):
        print("\n[Step 1] Generating 3 Top-K tables...")
        gen_topk_tables(args.out_dir, 'csv')

    # ---- Draw combined figures ----
    if args.step in ('figures','all'):
        print("\n[Step 2] Drawing 4×7 combined figures...")
        all_data = load_all()
        print(f"  Loaded {len(all_data)}/28 datasets")
        draw_combined(all_data, args.out_dir, args.fmt)

    print(f"\n{'='*60}")
    print(f"  [DONE] Output: {args.out_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
