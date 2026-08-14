#!/usr/bin/env python3
"""
Draw 4 combined figures (F1 / Precision / Recall / ROC) for 24 datasets.
Layout: 4 rows × 6 columns = 24 subplots.
- No suptitle, no global axis labels
- Dataset name below each subplot (very large font)
- Single legend at the bottom of the entire figure (very large font)
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

# ── 24 datasets (removed: abalone, arrhythmia, bank, hepatitis, covertype, skin) ──
DATASETS_24 = [
    "adult", "bank-full", "banknote", "breast-cancer", "car", "chess",
    "cmc", "credit", "diabetes", "german", "glass", "heart",
    "horse", "iris", "mushroom", "nursery", "parkinsons", "raisin",
    "student-mat", "wine", "wine-red", "wine-white", "yeast", "zoo",
]

# ── Algorithms & colours ──
ALGOS = ['ADFNR', 'DASOD', 'GCN', 'GCN-LOF', 'NIEOD', 'NMIGOD']
ALGO_COLORS = {
    'ADFNR': '#1f77b4', 'DASOD': '#ff7f0e', 'GCN': '#2ca02c',
    'GCN-LOF': '#9467bd', 'NIEOD': '#17becf', 'NMIGOD': '#E31818',
}
ALGO_LW = {a: (3.5 if a == 'NMIGOD' else 1.8) for a in ALGOS}
ALGO_ALPHA = {a: (0.95 if a == 'NMIGOD' else 0.65) for a in ALGOS}
ALGO_Z = {a: (10 if a == 'NMIGOD' else 5) for a in ALGOS}

# ── Font sizes (very large) ──
SUBTITLE_FONT = 24       # dataset name below each subplot
TICK_FONT = 16           # axis tick labels
LEGEND_FONT = 22         # legend entries
XLABEL_FONT = 28         # x-axis label (only kept per subplot = dataset name)
GLOBAL_LABEL_FONT = 24   # not used (removed)


def load_one(csv_path):
    """Load detection_results.csv, return sorted scores and binarized actuals."""
    df = pd.read_csv(csv_path)
    raw_scores = df.iloc[:, 1].astype(float).values
    raw_actuals = df.iloc[:, 3].astype(float).values
    idx = np.argsort(-raw_scores)
    scores = raw_scores[idx]
    actuals = (raw_actuals[idx] > 0).astype(int)
    return scores, actuals


def compute_prf(scores, actuals):
    n = len(scores)
    total = int(actuals.sum())
    if total == 0:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    tp = np.cumsum(actuals)
    k = np.arange(1, n + 1, dtype=np.float64)
    prec = tp / k
    rec = tp / total
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-15)
    return prec, rec, f1


def downsample(arr, max_pts=200):
    n = len(arr)
    if n <= max_pts:
        return np.arange(n), arr
    step = max(1, n // max_pts)
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx, arr[idx]


def load_all():
    """Load detection results for all 24 datasets × 6 algorithms."""
    all_data = {}
    for ds in DATASETS_24:
        ds_data = {}
        for algo in ALGOS:
            p = PROJECT_ROOT / algo / f'output_{ds}' / 'detection_results.csv'
            if not p.exists():
                continue
            sc, ac = load_one(str(p))
            if sc is None:
                continue
            prec, rec, f1 = compute_prf(sc, ac)
            fpr, tpr, _ = roc_curve(ac, sc)
            roc_auc = auc(fpr, tpr)
            ds_data[algo] = {
                'prec': prec, 'rec': rec, 'f1': f1,
                'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'n': len(sc),
            }
        if ds_data:
            all_data[ds] = ds_data
    return all_data


# =====================================================================
# DRAW COMBINED 4×6 FIGURES
# =====================================================================
def draw_combined(all_data, out_dir, fmt):
    n_rows, n_cols = 4, 6  # 24 subplots

    metric_configs = [
        {'key': 'f1',  'ylabel': 'F1-Score',   'fname': 'combined_4x6_f1'},
        {'key': 'prec','ylabel': 'Precision',   'fname': 'combined_4x6_precision'},
        {'key': 'rec', 'ylabel': 'Recall',      'fname': 'combined_4x6_recall'},
        {'key': 'roc', 'ylabel': 'True Positive Rate', 'fname': 'combined_4x6_roc'},
    ]

    for cfg in metric_configs:
        key = cfg['key']
        ylabel = cfg['ylabel']
        fname = cfg['fname']

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(36, 24))
        axes = axes.flatten()

        for idx, ds_name in enumerate(DATASETS_24):
            ax = axes[idx]

            if ds_name not in all_data:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=18, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel(ds_name, fontsize=SUBTITLE_FONT, fontweight='bold', labelpad=6)
                continue

            ds_data = all_data[ds_name]
            algos_present = [a for a in ALGOS if a in ds_data]

            if key == 'roc':
                for algo in algos_present:
                    d = ds_data[algo]
                    idx_p, fpr_p = downsample(d['fpr'])
                    tpr_p = d['tpr'][idx_p]
                    ax.plot(fpr_p, tpr_p,
                            label=f"{algo} ({d['auc']:.3f})",
                            color=ALGO_COLORS[algo],
                            linewidth=ALGO_LW[algo],
                            alpha=ALGO_ALPHA[algo],
                            zorder=ALGO_Z[algo])
                ax.plot([0, 1], [0, 1], 'k:', linewidth=0.6, alpha=0.3)
            else:
                for algo in algos_present:
                    d = ds_data[algo]
                    y_vals = d[key]
                    k_norm = np.linspace(0, 1, len(y_vals))
                    idx_p, y_p = downsample(y_vals)
                    ax.plot(k_norm[idx_p], y_p, label=algo,
                            color=ALGO_COLORS[algo],
                            linewidth=ALGO_LW[algo],
                            alpha=ALGO_ALPHA[algo],
                            zorder=ALGO_Z[algo])
                ax.set_ylim(0, 1.05)
                ax.set_yticks([0, 0.5, 1])

            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(labelsize=TICK_FONT)
            ax.grid(True, linestyle='--', alpha=0.3)

            # ── Dataset name below each subplot (very large) ──
            ax.set_xlabel(ds_name, fontsize=SUBTITLE_FONT, fontweight='bold', labelpad=8)

        # Hide unused axes (should be none with exactly 24)
        for idx in range(len(DATASETS_24), n_rows * n_cols):
            axes[idx].set_visible(False)

        # ── NO suptitle, NO global axis labels ──

        # ── Single common legend at the very bottom ──
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            # Deduplicate while preserving order
            seen = set()
            unique_hl = []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    unique_hl.append((h, l))
            handles_u, labels_u = zip(*unique_hl)

            leg = fig.legend(handles_u, labels_u, loc='lower center',
                             ncol=6, fontsize=LEGEND_FONT, framealpha=0.95,
                             bbox_to_anchor=(0.5, -0.02), borderpad=0.8,
                             handlelength=2.5, handletextpad=0.6,
                             columnspacing=1.5)

        # Tight layout with space at bottom for legend
        plt.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.10,
                            wspace=0.25, hspace=0.45)

        out_path = os.path.join(out_dir, f'{fname}.{fmt}')
        fig.savefig(out_path, dpi=180, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  [{key}] -> {out_path}")


# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', '-o', default=str(IMAGES_ROOT))
    parser.add_argument('--fmt', '-f', default='svg')
    args = parser.parse_args()

    out_dir = args.out_dir
    fmt = args.fmt

    print(f"{'='*60}")
    print(f"  24 datasets (removed: abalone, arrhythmia, bank, hepatitis, covertype, skin)")
    print(f"  Layout: 4 rows × 6 columns")
    print(f"  Fonts: subtitle={SUBTITLE_FONT}, tick={TICK_FONT}, legend={LEGEND_FONT}")
    print(f"  Output format: {fmt}")
    print(f"{'='*60}")

    print("\nLoading detection results...")
    all_data = load_all()
    print(f"  Loaded {len(all_data)}/24 datasets")

    print("\nDrawing combined 4×6 figures...")
    draw_combined(all_data, out_dir, fmt)

    print(f"\n{'='*60}")
    print(f"  [DONE] Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
