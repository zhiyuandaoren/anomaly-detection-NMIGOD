"""
Statistical test script for anomaly detection algorithms.
Performs Friedman test + Nemenyi post-hoc and generates CD diagrams
for both F1-Score and AUC.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import os

# ============================================================
# Configuration
# ============================================================

ALGORITHMS = ['ADFNR', 'DASOD', 'GCN', 'GCN-LOF', 'NIEOD', 'NMIGOD']
ALPHA = 0.05

# Critical values for the two-tailed Nemenyi test (q-values)
# q_alpha for alpha=0.05, k=2..10 algorithms
Q_ALPHA_TABLE = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Data Loading
# ============================================================

def load_data():
    """Load F1 and AUC data from the metrics directory."""
    base = os.path.join(os.path.dirname(OUTPUT_DIR), 'metrics')

    f1_df = pd.read_csv(os.path.join(base, 'f1_score.csv'), index_col=0)
    auc_df = pd.read_csv(os.path.join(base, 'auc.csv'), index_col=0)

    # Remove 'Average' row if present
    if 'Average' in f1_df.index:
        f1_df = f1_df.drop('Average')
    if 'Average' in auc_df.index:
        auc_df = auc_df.drop('Average')

    return f1_df, auc_df


# ============================================================
# Statistical Tests
# ============================================================

def compute_ranks(data_df):
    """
    Compute average ranks for each algorithm across all datasets.
    For each dataset, rank 1 = best (highest value), higher rank = worse.
    Returns: dict of algorithm -> average rank
    """
    n_datasets, n_algorithms = data_df.shape
    ranks = np.zeros((n_datasets, n_algorithms))

    for i in range(n_datasets):
        # Higher value = better, so use negative for ranking (rank 1 = best)
        row = data_df.iloc[i].values
        # argsort ascending: smallest value gets rank 1
        # We want largest value to get rank 1, so negate
        rank_order = np.argsort(np.argsort(-row))
        ranks[i] = rank_order + 1  # ranks are 1-indexed

    avg_ranks = {}
    for j, algo in enumerate(data_df.columns):
        avg_ranks[algo] = np.mean(ranks[:, j])

    return avg_ranks, ranks


def friedman_test(ranks_matrix):
    """
    Perform Friedman test.
    ranks_matrix: (n_datasets, n_algorithms) array of per-dataset ranks.

    Returns: statistic, p_value
    """
    n, k = ranks_matrix.shape
    # Average rank of each algorithm
    R_j = np.mean(ranks_matrix, axis=0)

    # Friedman statistic
    chi2_F = (12 * n) / (k * (k + 1)) * np.sum((R_j - (k + 1) / 2) ** 2)

    # Correction for ties - simplified (assuming few ties)
    # Full tie correction would require computing rank counts per dataset

    # Actually, use scipy's implementation for correctness
    # Reconstruct the data: we have per-dataset ranks, but scipy needs raw scores
    # Let's use the ranks-based formula directly
    statistic = chi2_F
    p_value = 1 - stats.chi2.cdf(chi2_F, df=k - 1)

    return statistic, p_value


def friedman_test_from_data(data_df):
    """
    Perform Friedman test directly from raw data using scipy.
    """
    n_datasets, n_algorithms = data_df.shape

    # scipy.stats.friedmanchisquare expects each algorithm's scores as a separate argument
    algo_data = [data_df[algo].values for algo in data_df.columns]
    statistic, p_value = stats.friedmanchisquare(*algo_data)

    return statistic, p_value


def nemenyi_cd(k, n, alpha=0.05):
    """
    Compute the Nemenyi Critical Difference.

    Parameters:
        k: number of algorithms
        n: number of datasets
        alpha: significance level

    Returns:
        CD: critical difference value
    """
    q_alpha = Q_ALPHA_TABLE.get(k, 3.164)  # default to k=10 value
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    return cd, q_alpha


# ============================================================
# CD Diagram Plotting
# ============================================================

def draw_cd_diagram(avg_ranks, cd, metric_name, friedman_stat, friedman_p, filename):
    """
    Draw a Critical Difference diagram.

    Parameters:
        avg_ranks: dict of algorithm -> average rank
        cd: Nemenyi critical difference
        metric_name: 'F1-Score' or 'AUC'
        friedman_stat: Friedman statistic
        friedman_p: Friedman p-value
        filename: output PNG filename
    """
    # Sort algorithms by average rank (ascending = better)
    sorted_algos = sorted(avg_ranks.items(), key=lambda x: x[1])
    algo_names = [a for a, r in sorted_algos]
    ranks = np.array([r for a, r in sorted_algos])

    k = len(algo_names)
    n_colors = k

    # Color palette - generate distinct colors
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(k)]

    # Figure setup
    fig, ax = plt.subplots(figsize=(14, 5))

    # Rank axis (reversed: rank 1 on the right, higher on the left)
    # We plot on a horizontal axis where smaller rank = better (right side)
    rank_min = 1
    rank_max = k

    # Add padding
    padding = 0.8
    x_min = rank_min - padding
    x_max = rank_max + padding

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, 3.5)

    # Draw the rank axis line
    ax.axhline(y=2.0, xmin=0.05, xmax=0.95, color='black', linewidth=1.5, zorder=1)

    # Draw tick marks and rank numbers
    for r in range(1, k + 1):
        ax.axvline(x=r, ymin=0.42, ymax=0.58, color='black', linewidth=1)
        ax.text(r, 1.35, str(r), ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw algorithm markers
    for i, (name, rank) in enumerate(zip(algo_names, ranks)):
        color = colors[i]
        ax.plot(rank, 2.0, 'o', markersize=16, color=color,
                markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        # Algorithm name above
        ax.text(rank, 2.6, name, ha='center', va='bottom', fontsize=13,
                fontweight='bold', color=color)

    # Draw CD bars connecting non-significantly different algorithms
    cd_bar_y = 1.5
    for i in range(k):
        rank_i = ranks[i]
        for j in range(i + 1, k):
            rank_j = ranks[j]
            diff = abs(rank_i - rank_j)
            if diff < cd:
                # These algorithms are NOT significantly different
                ax.plot([rank_i, rank_j], [cd_bar_y, cd_bar_y],
                        'k-', linewidth=2, zorder=2)
                # Draw small vertical ticks at endpoints
                ax.plot([rank_i, rank_i], [cd_bar_y - 0.1, cd_bar_y + 0.1],
                        'k-', linewidth=1.5)
                ax.plot([rank_j, rank_j], [cd_bar_y - 0.1, cd_bar_y + 0.1],
                        'k-', linewidth=1.5)

    # Annotate CD value
    ax.text(x_max - 0.2, 1.0, f'CD = {cd:.3f}', ha='right', va='center',
            fontsize=11, style='italic', color='gray')

    # Title and labels
    significance = 'significant' if friedman_p < ALPHA else 'not significant'
    ax.set_title(
        f'Critical Difference Diagram — {metric_name}\n'
        f'Friedman $\\chi^2$ = {friedman_stat:.3f}, '
        f'$p$ = {friedman_p:.4f} ({significance}, $\\alpha$ = {ALPHA})',
        fontsize=14, fontweight='bold', pad=18
    )

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add "Better →" and "← Worse" labels
    ax.text(x_max, 1.8, 'Better →', ha='right', va='center',
            fontsize=9, style='italic', color='gray')
    ax.text(x_min, 1.8, '← Worse', ha='left', va='center',
            fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"CD diagram saved: {filepath}")

    return filepath


def draw_combined_cd_diagram(f1_ranks, auc_ranks, f1_cd, auc_cd,
                              f1_stat, f1_p, auc_stat, auc_p, filename):
    """Draw a combined figure with F1 and AUC CD diagrams side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 5.5))

    def _draw_single(ax, avg_ranks, cd, metric_name, friedman_stat, friedman_p):
        sorted_algos = sorted(avg_ranks.items(), key=lambda x: x[1])
        algo_names = [a for a, r in sorted_algos]
        ranks = np.array([r for a, r in sorted_algos])
        k = len(algo_names)
        cmap = plt.cm.tab10
        colors = [cmap(i % 10) for i in range(k)]

        padding = 0.8
        rank_min = 1
        rank_max = k
        x_min = rank_min - padding
        x_max = rank_max + padding

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, 3.5)

        ax.axhline(y=2.0, xmin=0.05, xmax=0.95, color='black', linewidth=1.5, zorder=1)

        for r in range(1, k + 1):
            ax.axvline(x=r, ymin=0.42, ymax=0.58, color='black', linewidth=1)
            ax.text(r, 1.35, str(r), ha='center', va='center', fontsize=11, fontweight='bold')

        for i, (name, rank) in enumerate(zip(algo_names, ranks)):
            color = colors[i]
            ax.plot(rank, 2.0, 'o', markersize=14, color=color,
                    markeredgecolor='white', markeredgewidth=1.2, zorder=3)
            ax.text(rank, 2.6, name, ha='center', va='bottom', fontsize=11,
                    fontweight='bold', color=color)

        cd_bar_y = 1.5
        for i in range(k):
            rank_i = ranks[i]
            for j in range(i + 1, k):
                rank_j = ranks[j]
                if abs(rank_i - rank_j) < cd:
                    ax.plot([rank_i, rank_j], [cd_bar_y, cd_bar_y],
                            'k-', linewidth=2, zorder=2)
                    ax.plot([rank_i, rank_i], [cd_bar_y - 0.08, cd_bar_y + 0.08],
                            'k-', linewidth=1.2)
                    ax.plot([rank_j, rank_j], [cd_bar_y - 0.08, cd_bar_y + 0.08],
                            'k-', linewidth=1.2)

        ax.text(x_max - 0.2, 1.0, f'CD = {cd:.3f}', ha='right', va='center',
                fontsize=10, style='italic', color='gray')

        significance = 'significant' if friedman_p < ALPHA else 'not significant'
        ax.set_title(
            f'{metric_name}\n'
            f'Friedman $\\chi^2$ = {friedman_stat:.3f}, $p$ = {friedman_p:.4f}',
            fontsize=13, fontweight='bold', pad=12
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(x_max, 1.8, 'Better →', ha='right', va='center',
                fontsize=8, style='italic', color='gray')
        ax.text(x_min, 1.8, '← Worse', ha='left', va='center',
                fontsize=8, style='italic', color='gray')

    _draw_single(ax1, f1_ranks, f1_cd, 'F1-Score', f1_stat, f1_p)
    _draw_single(ax2, auc_ranks, auc_cd, 'AUC', auc_stat, auc_p)

    fig.suptitle('Critical Difference Diagrams — Anomaly Detection Algorithms',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Combined CD diagram saved: {filepath}")
    return filepath


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Statistical Tests for Anomaly Detection Algorithms")
    print("=" * 60)

    # Load data
    f1_df, auc_df = load_data()
    n_datasets, n_algorithms = f1_df.shape
    print(f"\nDatasets: {n_datasets}")
    print(f"Algorithms: {', '.join(f1_df.columns)}")

    results = {}

    for metric_name, data_df in [('F1-Score', f1_df), ('AUC', auc_df)]:
        print(f"\n{'─' * 40}")
        print(f"  {metric_name}")
        print(f"{'─' * 40}")

        # Compute ranks
        avg_ranks, ranks_matrix = compute_ranks(data_df)

        print("\nAverage Ranks:")
        for algo, rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
            print(f"  {algo:12s}: {rank:.4f}")

        # Friedman test
        friedman_stat, friedman_p = friedman_test_from_data(data_df)

        print(f"\nFriedman Test:")
        print(f"  χ² = {friedman_stat:.4f}")
        print(f"  p  = {friedman_p:.6f}")
        print(f"  {'Significant' if friedman_p < ALPHA else 'Not significant'} at α = {ALPHA}")

        # Nemenyi CD
        k = n_algorithms
        n = n_datasets
        cd, q_alpha = nemenyi_cd(k, n, ALPHA)

        print(f"\nNemenyi Post-hoc Test:")
        print(f"  k = {k} algorithms, N = {n} datasets")
        print(f"  q_α({k}, ∞) = {q_alpha:.4f}")
        print(f"  CD = {cd:.4f}")

        # Store results
        results[metric_name] = {
            'avg_ranks': avg_ranks,
            'ranks_matrix': ranks_matrix,
            'friedman_stat': friedman_stat,
            'friedman_p': friedman_p,
            'nemenyi_cd': cd,
            'q_alpha': q_alpha,
        }

    # Draw individual CD diagrams
    f1_r = results['F1-Score']
    auc_r = results['AUC']

    draw_cd_diagram(
        f1_r['avg_ranks'], f1_r['nemenyi_cd'],
        'F1-Score', f1_r['friedman_stat'], f1_r['friedman_p'],
        'cd_diagram_f1.png'
    )

    draw_cd_diagram(
        auc_r['avg_ranks'], auc_r['nemenyi_cd'],
        'AUC', auc_r['friedman_stat'], auc_r['friedman_p'],
        'cd_diagram_auc.png'
    )

    # Draw combined CD diagram
    draw_combined_cd_diagram(
        f1_r['avg_ranks'], auc_r['avg_ranks'],
        f1_r['nemenyi_cd'], auc_r['nemenyi_cd'],
        f1_r['friedman_stat'], f1_r['friedman_p'],
        auc_r['friedman_stat'], auc_r['friedman_p'],
        'cd_diagram_combined.png'
    )

    # Save results to JSON
    output_json = {
        'n_datasets': n_datasets,
        'n_algorithms': n_algorithms,
        'alpha': ALPHA,
        'F1-Score': {
            'friedman_stat': round(f1_r['friedman_stat'], 6),
            'friedman_p': round(f1_r['friedman_p'], 6),
            'nemenyi_cd': round(f1_r['nemenyi_cd'], 6),
            'q_alpha': round(f1_r['q_alpha'], 6),
            'avg_ranks': {k: round(v, 4) for k, v in f1_r['avg_ranks'].items()},
            'significant': f1_r['friedman_p'] < ALPHA,
        },
        'AUC': {
            'friedman_stat': round(auc_r['friedman_stat'], 6),
            'friedman_p': round(auc_r['friedman_p'], 6),
            'nemenyi_cd': round(auc_r['nemenyi_cd'], 6),
            'q_alpha': round(auc_r['q_alpha'], 6),
            'avg_ranks': {k: round(v, 4) for k, v in auc_r['avg_ranks'].items()},
            'significant': auc_r['friedman_p'] < ALPHA,
        },
    }
    json_path = os.path.join(OUTPUT_DIR, 'statistical_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate per-dataset ranks table
    _save_rank_tables(f1_df, auc_df)

    print("\nDone!")


def _save_rank_tables(f1_df, auc_df):
    """Save per-dataset rank tables as CSV."""
    n_datasets = len(f1_df)

    # F1 ranks
    f1_ranks = np.zeros_like(f1_df.values, dtype=float)
    for i in range(n_datasets):
        row = f1_df.iloc[i].values
        f1_ranks[i] = np.argsort(np.argsort(-row)) + 1
    f1_rank_df = pd.DataFrame(f1_ranks, index=f1_df.index, columns=f1_df.columns)
    f1_rank_df.to_csv(os.path.join(OUTPUT_DIR, 'f1_ranks.csv'))
    print(f"F1 ranks saved to: {os.path.join(OUTPUT_DIR, 'f1_ranks.csv')}")

    # AUC ranks
    auc_ranks = np.zeros_like(auc_df.values, dtype=float)
    for i in range(n_datasets):
        row = auc_df.iloc[i].values
        auc_ranks[i] = np.argsort(np.argsort(-row)) + 1
    auc_rank_df = pd.DataFrame(auc_ranks, index=auc_df.index, columns=auc_df.columns)
    auc_rank_df.to_csv(os.path.join(OUTPUT_DIR, 'auc_ranks.csv'))
    print(f"AUC ranks saved to: {os.path.join(OUTPUT_DIR, 'auc_ranks.csv')}")


if __name__ == '__main__':
    main()
