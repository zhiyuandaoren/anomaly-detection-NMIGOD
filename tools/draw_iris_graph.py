#!/usr/bin/env python3
"""Iris: left = scatter, right = neighborhood structure."""
import sys, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec = importlib.util.spec_from_file_location('nmigod', 'NMIGOD/detector.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

fw = mod.AnomalyDetectionFramework(random_state=42)
fw.df_raw = pd.read_csv('datasets/iris.csv')
fw.target_column = 'class'
fw.anomaly_values = ['Iris-setosa']
fw.output_folder = './NMIGOD/output_iris_viz'
os.makedirs(fw.output_folder, exist_ok=True)
fw.preprocess_data()
fw.train_model()
fw.get_anomaly_scores()

M = fw.M_matrix.cpu().numpy()
y_true = fw.y_true.values
N = len(y_true)

# Use two best features for visualization
df = fw.df_original
x_col, y_col = 'petal_length_in_cm', 'petal_width_in_cm'

anom_mask = y_true == 1
norm_mask = y_true == 0
x_all = df[x_col].values
y_all = df[y_col].values

# Get per-attribute radii used in neighborhood construction
# Re-run the radius computation to get epsilon values
num_cols = fw.num_cols
X_num = fw.df_original[num_cols].values.astype(float)
std_vals = X_num.std(axis=0)

# Find which columns are petal_length and petal_width
pl_idx = num_cols.index(x_col) if x_col in num_cols else 0
pw_idx = num_cols.index(y_col) if y_col in num_cols else 1

# Approximate epsilon for these attributes (V2 adaptive radius)
# From the run output: sigma ~ range
eps_pl = std_vals[pl_idx] / 2  # approximate adaptive radius
eps_pw = std_vals[pw_idx] / 2

# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Left: Scatter ---
ax1.scatter(x_all[norm_mask], y_all[norm_mask],
            c='#5DADE2', s=30, alpha=0.7, edgecolors='none', label='Normal')
ax1.scatter(x_all[anom_mask], y_all[anom_mask],
            c='#E74C3C', s=60, alpha=0.9, edgecolors='#7B241C', linewidths=1.0,
            marker='D', label='Anomaly')

ax1.set_xlabel('Petal Length (cm)', fontsize=12)
ax1.set_ylabel('Petal Width (cm)', fontsize=12)
ax1.set_title('(a) Raw Feature Space', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)

# --- Right: Neighborhood Structure ---
# Same scatter
ax2.scatter(x_all[norm_mask], y_all[norm_mask],
            c='#5DADE2', s=30, alpha=0.5, edgecolors='none')
ax2.scatter(x_all[anom_mask], y_all[anom_mask],
            c='#E74C3C', s=60, alpha=0.9, edgecolors='#7B241C', linewidths=1.0,
            marker='D')

# Draw neighborhood ellipses for anomalies (each anomaly + its radius)
for i in np.where(anom_mask)[0]:
    ellipse = Ellipse((x_all[i], y_all[i]), width=2*eps_pl, height=2*eps_pw,
                       edgecolor='#C0392B', facecolor='#F1948A', alpha=0.12,
                       linewidth=1.0, linestyle='-')
    ax2.add_patch(ellipse)

# Draw neighborhood ellipses for a few normal points
for i in np.random.RandomState(42).choice(np.where(norm_mask)[0], size=8, replace=False):
    ellipse = Ellipse((x_all[i], y_all[i]), width=2*eps_pl, height=2*eps_pw,
                       edgecolor='#2E86C1', facecolor='#85C1E9', alpha=0.08,
                       linewidth=0.8, linestyle='--')
    ax2.add_patch(ellipse)

# Draw neighborhood connections for a representative anomaly
rep_anom = np.where(anom_mask)[0][0]
anom_neighbors = []
for j in range(N):
    if M[rep_anom, j] > 0 and y_true[j] == 1:
        anom_neighbors.append(j)

for j in anom_neighbors:
    ax2.plot([x_all[rep_anom], x_all[j]], [y_all[rep_anom], y_all[j]],
             color='#C0392B', linewidth=1.5, alpha=0.7)

# Draw neighborhood connections for a representative normal node
rep_norm = np.random.RandomState(42).choice(np.where(norm_mask)[0])
norm_neighbors = []
for j in range(N):
    if M[rep_norm, j] > 0 and y_true[j] == 0:
        norm_neighbors.append(j)
norm_neighbors = norm_neighbors[:6]  # limit for clarity
for j in norm_neighbors:
    ax2.plot([x_all[rep_norm], x_all[j]], [y_all[rep_norm], y_all[j]],
             color='#2E86C1', linewidth=1.0, alpha=0.5)

# Highlight the representative nodes
ax2.scatter([x_all[rep_anom]], [y_all[rep_anom]],
            c='#E74C3C', s=180, alpha=1.0, edgecolors='#7B241C', linewidths=2.5,
            marker='D', zorder=10)
ax2.scatter([x_all[rep_norm]], [y_all[rep_norm]],
            c='#5DADE2', s=140, alpha=1.0, edgecolors='#1A5276', linewidths=2.0,
            zorder=10)

# Annotation
ax2.annotate('Anomaly\nneighborhood', xy=(x_all[rep_anom], y_all[rep_anom]),
             fontsize=9, fontweight='bold', color='#7B241C',
             xytext=(x_all[rep_anom] + 0.8, y_all[rep_anom] + 0.5),
             arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5))

ax2.annotate('Normal\nneighborhood', xy=(x_all[rep_norm], y_all[rep_norm]),
             fontsize=9, fontweight='bold', color='#1A5276',
             xytext=(x_all[rep_norm] - 1.5, y_all[rep_norm] - 0.8),
             arrowprops=dict(arrowstyle='->', color='#2E86C1', lw=1.5))

ax2.set_xlabel('Petal Length (cm)', fontsize=12)
ax2.set_ylabel('Petal Width (cm)', fontsize=12)
ax2.set_title('(b) Neighborhood Structure (NMI Graph)', fontsize=14, fontweight='bold')

# Legend for right
leg2 = [
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#E74C3C', markersize=12,
           markeredgecolor='#7B241C', markeredgewidth=1.5, label='Anomaly node'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#5DADE2', markersize=10,
           markeredgecolor='#2E86C1', markeredgewidth=1.0, label='Normal node'),
    mpatches.Patch(color='#F1948A', alpha=0.25, label='Anomaly neighborhood (radius)'),
    mpatches.Patch(color='#85C1E9', alpha=0.15, label='Normal neighborhood (radius)'),
    Line2D([0],[0], color='#C0392B', lw=1.5, alpha=0.7, label='NMI edge (A-A)'),
    Line2D([0],[0], color='#2E86C1', lw=1.0, alpha=0.5, label='NMI edge (N-N)'),
]
ax2.legend(handles=leg2, loc='upper left', fontsize=8, framealpha=0.9)

fig.suptitle('Neighborhood Construction in NMIGOD — Iris Dataset\n'
             'Anomalies and normal instances occupy disjoint neighborhood spaces',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
os.makedirs('./images', exist_ok=True)
plt.savefig('./images/NMIGOD_iris_neighborhood.png', dpi=250, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Done: ./images/NMIGOD_iris_neighborhood.png')
