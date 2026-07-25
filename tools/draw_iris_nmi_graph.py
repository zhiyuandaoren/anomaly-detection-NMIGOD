#!/usr/bin/env python3
"""NMI graph schematic — paper-style, matching img.png from 0714.docx."""
import sys, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import networkx as nx

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

anom_idx = np.array([4, 0, 7])
norm_idx = np.array([48, 31, 66, 33, 37])
sel = np.concatenate([anom_idx, norm_idx])

G = nx.Graph()
G.add_nodes_from(sel)
for i, a in enumerate(sel):
    for b in sel[i+1:]:
        w = M[a, b]
        if w > 0:
            G.add_edge(a, b, weight=round(float(w), 2))

a_a = [(u,v) for u,v in G.edges() if y_true[u]==1 and y_true[v]==1]
a_n = [(u,v) for u,v in G.edges() if (y_true[u]==1) ^ (y_true[v]==1)]
n_n = [(u,v) for u,v in G.edges() if y_true[u]==0 and y_true[v]==0]

pos = {
    4:  np.array([-1.5, 1.0]),
    0:  np.array([-2.5, -0.5]),
    7:  np.array([-0.5, -0.5]),
    48: np.array([3.0, 2.0]),
    31: np.array([4.5, 0.5]),
    66: np.array([3.0, -1.5]),
    33: np.array([5.5, 2.0]),
    37: np.array([5.5, -1.0]),
}

fig, ax = plt.subplots(figsize=(10, 5.5))

nx.draw_networkx_edges(G, pos, edgelist=n_n, ax=ax,
                       alpha=0.60, edge_color='#666666', width=1.2)
nx.draw_networkx_edges(G, pos, edgelist=a_n, ax=ax,
                       alpha=0.55, edge_color='#E67E22', width=1.2, style='dashed')
nx.draw_networkx_edges(G, pos, edgelist=a_a, ax=ax,
                       alpha=0.85, edge_color='#C0392B', width=2.2)

ew = {(u,v): str(d['weight']) for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=ew, ax=ax,
                             font_size=7.5, font_color='#444444',
                             label_pos=0.52,
                             bbox=dict(boxstyle='round,pad=0.08',
                                       facecolor='white', edgecolor='none', alpha=0.75))

nx.draw_networkx_nodes(G, pos, nodelist=anom_idx.tolist(), ax=ax,
                       node_color='#E74C3C', node_size=550, alpha=0.95,
                       edgecolors='#7B241C', linewidths=2.2, node_shape='D')
nx.draw_networkx_nodes(G, pos, nodelist=norm_idx.tolist(), ax=ax,
                       node_color='#5DADE2', node_size=550, alpha=0.85,
                       edgecolors='#2E86C1', linewidths=1.5)

labels = {}
for idx in sel:
    kind = '(A)' if y_true[idx] == 1 else '(N)'
    labels[idx] = f'$x_{{{idx}}}$ {kind}'
nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8.5, font_weight='bold')

ax.annotate('Component 1\n(Anomaly)', xy=(-1.5, 1.5), fontsize=10,
            fontweight='bold', color='#7B241C', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8',
                      edgecolor='#E74C3C', alpha=0.85))
ax.annotate('Component 2\n(Normal)', xy=(4.5, 3.0), fontsize=10,
            fontweight='bold', color='#2E86C1', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6EAF8',
                      edgecolor='#5DADE2', alpha=0.85))

leg = [mpatches.Patch(color='#E74C3C', label='Anomaly node'),
       mpatches.Patch(color='#5DADE2', label='Normal node'),
       Line2D([0],[0], color='#C0392B', lw=2.2, label='A-A edge'),
       Line2D([0],[0], color='#666666', lw=1.2, label='N-N edge')]
ax.legend(handles=leg, loc='lower right', fontsize=9, framealpha=0.9,
          edgecolor='#cccccc')

ax.set_xlim(-4.0, 7.5)
ax.set_ylim(-3.0, 4.5)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

plt.tight_layout()
os.makedirs('./images', exist_ok=True)
plt.savefig('./images/NMIGOD_iris_graph.png', dpi=250, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Done: ./images/NMIGOD_iris_graph.png')
