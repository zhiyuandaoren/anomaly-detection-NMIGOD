"""
Generate CD (Critical Difference) diagrams — Demsar (2006) style.
Matches the reference CD diagram style:
  - CD scale bar at top (thick horizontal bar)
  - Rank axis with tick marks and rank numbers
  - Colored algorithm markers (red-blue palette)
  - Thick black connecting bars for non-significant groups
  - Algorithm names below axis in matching colors

Outputs SVG (primary) and PDF for both F1 and AUC.
"""

import sys, io, json, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ============================================================
# Config
# ============================================================

ALPHA = 0.05

# Nemenyi q-values (alpha=0.05)
Q_ALPHA_TABLE = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Color palette (red-blue family, matching reference style) ----
ALGO_COLORS = {
    'ADFNR':   '#c9243f',  # red
    'DASOD':   '#d4556b',  # lighter red
    'GCN':     '#1e5fa8',  # blue
    'GCN-LOF': '#2975c7',  # medium blue
    'NIEOD':   '#e0884a',  # orange-brown
    'NMIGOD':  '#1450a0',  # dark blue (best performer)
}

ALGO_MARKERS = {
    'ADFNR':   'o',
    'DASOD':   's',
    'GCN':     'D',
    'GCN-LOF': '^',
    'NIEOD':   'v',
    'NMIGOD':  'P',
}

# ============================================================
# Data
# ============================================================

def load_data():
    base = os.path.join(os.path.dirname(OUTPUT_DIR), 'metrics')
    f1_df = pd.read_csv(os.path.join(base, 'f1_score.csv'), index_col=0)
    auc_df = pd.read_csv(os.path.join(base, 'auc.csv'), index_col=0)
    for df in [f1_df, auc_df]:
        if 'Average' in df.index:
            df.drop('Average', inplace=True)
    return f1_df, auc_df


def compute_ranks(data_df):
    n = data_df.shape[0]
    ranks = np.zeros_like(data_df.values, dtype=float)
    for i in range(n):
        ranks[i] = np.argsort(np.argsort(-data_df.iloc[i].values)) + 1
    avg_ranks = {algo: np.mean(ranks[:, j])
                 for j, algo in enumerate(data_df.columns)}
    return avg_ranks, ranks


def friedman_test_from_data(data_df):
    algo_data = [data_df[algo].values for algo in data_df.columns]
    return stats.friedmanchisquare(*algo_data)


def nemenyi_cd(k, n, alpha=0.05):
    q_alpha = Q_ALPHA_TABLE.get(k, 3.164)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    return cd, q_alpha


# ============================================================
# CD Diagram — Reference Style
# ============================================================

def draw_cd_diagram(avg_ranks, cd, metric_name,
                    friedman_stat, friedman_p, filename_stem):
    """
    Single CD diagram in Demsar reference style.
    Output: SVG + PDF.
    """

    sorted_algos = sorted(avg_ranks.items(), key=lambda x: x[1])
    algo_names  = [a for a, r in sorted_algos]
    ranks       = np.array([r for a, r in sorted_algos])
    k           = len(algo_names)

    # -- figure --
    fig, ax = plt.subplots(figsize=(14, 5.0))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.82, bottom=0.28)

    rank_min, rank_max = 1, k
    pad = 0.85
    ax.set_xlim(rank_min - pad, rank_max + pad)
    ax.set_ylim(-1.2, 5.0)
    ax.set_aspect('auto')

    # ---- CD scale bar (top) ----
    bar_y    = 4.4
    cd_left  = rank_max - cd
    cd_right = rank_max

    # thick bar
    ax.plot([cd_left, cd_right], [bar_y, bar_y],
            'k-', linewidth=5, solid_capstyle='butt', zorder=5)
    # end caps
    for x in [cd_left, cd_right]:
        ax.plot([x, x], [bar_y - 0.22, bar_y + 0.22],
                'k-', linewidth=1.8, zorder=5)
    # label
    ax.text((cd_left + cd_right) / 2, bar_y + 0.52,
            f'CD = {cd:.2f}', ha='center', va='bottom',
            fontsize=11.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='gray',
                      alpha=0.9, linewidth=0.7))

    # ---- rank axis ----
    axis_y = 2.0
    ax.plot([rank_min, rank_max], [axis_y, axis_y],
            'k-', linewidth=1.5, zorder=2)

    # tick marks & rank numbers
    for r in range(1, k + 1):
        ax.plot([r, r], [axis_y - 0.25, axis_y + 0.25],
                'k-', linewidth=1.2, zorder=2)
        ax.text(r, axis_y - 0.58, str(r), ha='center', va='top',
                fontsize=15, fontweight='bold', color='#333333')

    # ---- CD connecting bars (below axis) ----
    conn_y = axis_y - 1.05
    # Build contiguous groups of algorithms not significantly different
    # Sort ranks, find groups where max rank - min rank < CD
    visited = [False] * k
    for i in range(k):
        if visited[i]:
            continue
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) < cd:
            j += 1
        if j > i:
            ax.plot([ranks[i], ranks[j]], [conn_y, conn_y],
                    'k-', linewidth=4.5, solid_capstyle='round', zorder=3)
            for t in range(i, j + 1):
                visited[t] = True
        else:
            visited[i] = True

    # ---- algorithm markers (on axis) ----
    for i, (name, rank) in enumerate(zip(algo_names, ranks)):
        color = ALGO_COLORS.get(name, '#555555')
        marker = ALGO_MARKERS.get(name, 'o')
        ax.plot(rank, axis_y, marker, markersize=14, color=color,
                markeredgecolor='white', markeredgewidth=1.5,
                zorder=6, clip_on=False)

    # ---- algorithm names (below) ----
    # Arrange with offsets to avoid overlap
    label_y_base = -0.15
    offsets = _compute_label_offsets(ranks, min_gap=0.55)
    for i, (name, rank) in enumerate(zip(algo_names, ranks)):
        color = ALGO_COLORS.get(name, '#555555')
        marker = ALGO_MARKERS.get(name, 'o')
        ly = label_y_base + offsets[i]
        # small colored marker
        ax.plot(rank, ly + 0.12, marker, markersize=8, color=color,
                markeredgecolor='white', markeredgewidth=0.8,
                zorder=6, clip_on=False)
        ax.text(rank, ly - 0.15, name, ha='center', va='top',
                fontsize=11, fontweight='bold', color=color)

    # ---- title ----
    sig = 'Significant' if friedman_p < ALPHA else 'Not significant'
    p_str = '$p < 0.0001$' if friedman_p < 0.0001 else f'$p = {friedman_p:.4f}$'
    title = (
        f'{metric_name} — Critical Difference Diagram\n'
        f'Friedman $\\chi^2_{{\\mathrm{{F}}}} = {friedman_stat:.3f}$,  '
        f'{p_str}  ({sig}, $\\alpha = {ALPHA}$)'
    )
    ax.set_title(title, fontsize=13.5, fontweight='bold', pad=18,
                 color='#222222')

    # ---- clean ----
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # ---- save ----
    for fmt, ext in [('svg', 'svg'), ('pdf', 'pdf')]:
        path = os.path.join(OUTPUT_DIR, f'{filename_stem}.{ext}')
        fig.savefig(path, format=fmt, facecolor='white', edgecolor='none',
                    bbox_inches='tight', pad_inches=0.15)
        print(f'  → {path}')
    plt.close(fig)


def _compute_label_offsets(ranks, min_gap=0.55):
    """Assign y offsets so overlapping labels don't collide."""
    k = len(ranks)
    offsets = np.zeros(k)
    for i in range(k):
        for j in range(i + 1, k):
            if abs(ranks[i] - ranks[j]) < min_gap:
                # they're close, stagger their offsets
                if offsets[i] == offsets[j]:
                    offsets[j] -= 0.52
    return offsets


# ============================================================
# Combined two-panel figure
# ============================================================

def draw_combined_cd(f1_ranks, auc_ranks, f1_cd, auc_cd,
                     f1_stat, f1_p, auc_stat, auc_p, filename_stem):
    """Side-by-side CD diagrams."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 5.2))
    fig.subplots_adjust(left=0.04, right=0.97, top=0.82, bottom=0.28, wspace=0.32)

    def _panel(ax, avg_ranks, cd, metric_name, friedman_stat, friedman_p):
        sorted_algos = sorted(avg_ranks.items(), key=lambda x: x[1])
        algo_names  = [a for a, r in sorted_algos]
        ranks       = np.array([r for a, r in sorted_algos])
        k           = len(algo_names)
        rank_min, rank_max = 1, k
        pad = 0.85
        ax.set_xlim(rank_min - pad, rank_max + pad)
        ax.set_ylim(-1.2, 5.0)

        # CD scale bar
        bar_y = 4.4
        cd_left = rank_max - cd
        cd_right = rank_max
        ax.plot([cd_left, cd_right], [bar_y, bar_y],
                'k-', linewidth=5, solid_capstyle='butt', zorder=5)
        for x in [cd_left, cd_right]:
            ax.plot([x, x], [bar_y - 0.18, bar_y + 0.18], 'k-', linewidth=1.5, zorder=5)
        ax.text((cd_left + cd_right) / 2, bar_y + 0.45,
                f'CD = {cd:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray',
                          alpha=0.9, linewidth=0.7))

        # rank axis
        axis_y = 2.0
        ax.plot([rank_min, rank_max], [axis_y, axis_y], 'k-', linewidth=1.5, zorder=2)
        for r in range(1, k + 1):
            ax.plot([r, r], [axis_y - 0.2, axis_y + 0.2], 'k-', linewidth=1.0, zorder=2)
            ax.text(r, axis_y - 0.52, str(r), ha='center', va='top',
                    fontsize=14, fontweight='bold', color='#333333')

        # connecting bars
        conn_y = axis_y - 1.05
        visited = [False] * k
        for i in range(k):
            if visited[i]:
                continue
            j = i
            while j + 1 < k and (ranks[j + 1] - ranks[i]) < cd:
                j += 1
            if j > i:
                ax.plot([ranks[i], ranks[j]], [conn_y, conn_y],
                        'k-', linewidth=4, solid_capstyle='round', zorder=3)
                for t in range(i, j + 1):
                    visited[t] = True
            else:
                visited[i] = True

        # markers
        for i, (name, rank) in enumerate(zip(algo_names, ranks)):
            color = ALGO_COLORS.get(name, '#555555')
            marker = ALGO_MARKERS.get(name, 'o')
            ax.plot(rank, axis_y, marker, markersize=13, color=color,
                    markeredgecolor='white', markeredgewidth=1.3, zorder=6)

        # labels
        label_y_base = -0.15
        offsets = _compute_label_offsets(ranks, min_gap=0.55)
        for i, (name, rank) in enumerate(zip(algo_names, ranks)):
            color = ALGO_COLORS.get(name, '#555555')
            marker = ALGO_MARKERS.get(name, 'o')
            ly = label_y_base + offsets[i]
            ax.plot(rank, ly + 0.12, marker, markersize=7, color=color,
                    markeredgecolor='white', markeredgewidth=0.7, zorder=6)
            ax.text(rank, ly - 0.15, name, ha='center', va='top',
                    fontsize=10.5, fontweight='bold', color=color)

        # title
        sig = 'Significant' if friedman_p < ALPHA else 'Not significant'
        p_str = '$p < 0.0001$' if friedman_p < 0.0001 else f'$p = {friedman_p:.4f}$'
        ax.set_title(
            f'{metric_name}\n'
            f'$\\chi^2_{{\\mathrm{{F}}}} = {friedman_stat:.3f}$,  {p_str}  ({sig})',
            fontsize=12, fontweight='bold', pad=16, color='#222222')

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    _panel(ax1, f1_ranks, f1_cd, 'F1-Score', f1_stat, f1_p)
    _panel(ax2, auc_ranks, auc_cd, 'AUC', auc_stat, auc_p)

    fig.suptitle('Critical Difference Diagrams — Anomaly Detection Algorithms',
                 fontsize=15, fontweight='bold', y=1.01, color='#222222')

    for fmt, ext in [('svg', 'svg'), ('pdf', 'pdf')]:
        path = os.path.join(OUTPUT_DIR, f'{filename_stem}.{ext}')
        fig.savefig(path, format=fmt, facecolor='white', edgecolor='none',
                    bbox_inches='tight', pad_inches=0.15)
        print(f'  → {path}')
    plt.close(fig)


# ============================================================
# Helpers
# ============================================================

def save_rank_tables(f1_df, auc_df):
    n = len(f1_df)
    f1_ranks = np.zeros_like(f1_df.values, dtype=float)
    for i in range(n):
        f1_ranks[i] = np.argsort(np.argsort(-f1_df.iloc[i].values)) + 1
    pd.DataFrame(f1_ranks, index=f1_df.index, columns=f1_df.columns) \
      .to_csv(os.path.join(OUTPUT_DIR, 'f1_ranks.csv'))
    print(f'F1 ranks → {os.path.join(OUTPUT_DIR, "f1_ranks.csv")}')

    auc_ranks = np.zeros_like(auc_df.values, dtype=float)
    for i in range(n):
        auc_ranks[i] = np.argsort(np.argsort(-auc_df.iloc[i].values)) + 1
    pd.DataFrame(auc_ranks, index=auc_df.index, columns=auc_df.columns) \
      .to_csv(os.path.join(OUTPUT_DIR, 'auc_ranks.csv'))
    print(f'AUC ranks → {os.path.join(OUTPUT_DIR, "auc_ranks.csv")}')


# ============================================================
# Main
# ============================================================

def main():
    print('=' * 60)
    print('CD Diagrams — Demsar Reference Style')
    print('=' * 60)

    f1_df, auc_df = load_data()
    n_datasets, n_algorithms = f1_df.shape
    k, n = n_algorithms, n_datasets
    print(f'\nDatasets: {n_datasets} | Algorithms: {k}')
    print(f'α = {ALPHA}')

    all_results = {}

    for metric, df in [('F1-Score', f1_df), ('AUC', auc_df)]:
        print(f'\n{"─" * 45}')
        print(f'  {metric}')
        print(f'{"─" * 45}')
        avg_ranks, _ = compute_ranks(df)
        friedman_stat, friedman_p = friedman_test_from_data(df)
        cd, q_alpha = nemenyi_cd(k, n, ALPHA)

        for algo, r in sorted(avg_ranks.items(), key=lambda x: x[1]):
            print(f'    {algo:10s}  {r:.3f}')
        print(f'  Friedman: χ²={friedman_stat:.4f}, p={friedman_p:.6f}')
        print(f'  Nemenyi:  CD={cd:.4f} (qα={q_alpha:.4f})')

        all_results[metric] = {
            'avg_ranks': avg_ranks,
            'friedman_stat': friedman_stat,
            'friedman_p': friedman_p,
            'nemenyi_cd': cd,
            'q_alpha': q_alpha,
        }

    # -- Draw --
    print(f'\n{"─" * 45}')
    print('  Drawing CD diagrams')
    print(f'{"─" * 45}')

    f1 = all_results['F1-Score']
    auc = all_results['AUC']

    print('\n[1/3] F1-Score')
    draw_cd_diagram(f1['avg_ranks'], f1['nemenyi_cd'],
                    'F1-Score', f1['friedman_stat'], f1['friedman_p'],
                    'cd_diagram_f1')

    print('\n[2/3] AUC')
    draw_cd_diagram(auc['avg_ranks'], auc['nemenyi_cd'],
                    'AUC', auc['friedman_stat'], auc['friedman_p'],
                    'cd_diagram_auc')

    print('\n[3/3] Combined')
    draw_combined_cd(f1['avg_ranks'], auc['avg_ranks'],
                     f1['nemenyi_cd'], auc['nemenyi_cd'],
                     f1['friedman_stat'], f1['friedman_p'],
                     auc['friedman_stat'], auc['friedman_p'],
                     'cd_diagram_combined')

    # -- JSON --
    json_out = {
        'n_datasets': n, 'n_algorithms': k, 'alpha': ALPHA,
        'F1-Score': {
            'friedman_stat': round(f1['friedman_stat'], 6),
            'friedman_p': round(f1['friedman_p'], 6),
            'nemenyi_cd': round(f1['nemenyi_cd'], 6),
            'q_alpha': round(f1['q_alpha'], 6),
            'avg_ranks': {k: round(v, 4) for k, v in f1['avg_ranks'].items()},
            'significant': bool(f1['friedman_p'] < ALPHA),
        },
        'AUC': {
            'friedman_stat': round(auc['friedman_stat'], 6),
            'friedman_p': round(auc['friedman_p'], 6),
            'nemenyi_cd': round(auc['nemenyi_cd'], 6),
            'q_alpha': round(auc['q_alpha'], 6),
            'avg_ranks': {k: round(v, 4) for k, v in auc['avg_ranks'].items()},
            'significant': bool(auc['friedman_p'] < ALPHA),
        },
    }
    jp = os.path.join(OUTPUT_DIR, 'statistical_test_results.json')
    with open(jp, 'w', encoding='utf-8') as fh:
        json.dump(json_out, fh, indent=2, ensure_ascii=False)
    print(f'\nJSON → {jp}')

    save_rank_tables(f1_df, auc_df)
    print('\n✅ Done!')


if __name__ == '__main__':
    main()
