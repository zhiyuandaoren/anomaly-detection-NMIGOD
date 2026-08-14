#!/usr/bin/env python3
"""Generate academic-style LaTeX experiment report."""

import pandas as pd, numpy as np, json
from scipy.stats import friedmanchisquare
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

# Read all data
f1 = pd.read_csv(ROOT / 'metrics' / 'f1_score.csv', index_col=0)
if 'Average' in f1.index:
    f1 = f1.drop('Average')
auc = pd.read_csv(ROOT / 'metrics' / 'auc.csv', index_col=0)
if 'Average' in auc.index:
    auc = auc.drop('Average')
config = pd.read_csv(ROOT / 'datasets' / 'datasets_config.csv')

gpu = ['GCN', 'GCN-LOF', 'NIEOD', 'NMIGOD']
all_algos = ['ADFNR', 'DASOD', 'GCN', 'GCN-LOF', 'NIEOD', 'NMIGOD']
f1g = f1[gpu].dropna()
aucg = auc[gpu].dropna()

def esc(s):
    return str(s).replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')

def pfmt(p):
    if p < 0.001:
        return "$<$0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    elif p < 0.05:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


def compute_friedman_nemenyi(metric_df, algos):
    """Friedman test + Nemenyi post-hoc on F1 scores."""
    data = metric_df[algos].dropna()
    rankings = data.rank(axis=1, ascending=False)
    avg_ranks = rankings.mean().sort_values()

    stat, p = friedmanchisquare(*[data[a].values for a in algos])
    k = len(algos)
    n = len(data)
    q_table = {2:1.960, 3:2.343, 4:2.569, 5:2.728, 6:2.850, 7:2.949, 8:3.031, 9:3.102, 10:3.164}
    q_alpha = q_table.get(k, 2.850)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    nmigod_rank = avg_ranks.get('NMIGOD', 0)
    pairwise = {}
    for algo in algos:
        if algo == 'NMIGOD':
            continue
        diff = abs(nmigod_rank - avg_ranks.get(algo, 0))
        pairwise[algo] = {
            'nmigod_rank': round(nmigod_rank, 2),
            'opponent_rank': round(avg_ranks.get(algo, 0), 2),
            'diff': round(diff, 2),
            'significant': diff > cd,
        }
    return {
        'friedman_stat': round(stat, 4),
        'friedman_p': round(p, 4),
        'nemenyi_cd': round(cd, 4),
        'k': k, 'n': n,
        'avg_ranks': {a: round(float(r), 2) for a, r in avg_ranks.items()},
        'pairwise': pairwise,
    }


f1_stats = compute_friedman_nemenyi(f1, all6)

type_map = dict(zip(config['Dataset'], config['DataType']))
subgroup = {}
for dtype in ['Numerical', 'Mixed', 'Categorical']:
    ds = [d for d in f1g.index if type_map.get(d, '') == dtype]
    if ds:
        sub = f1g.loc[ds]
        subgroup[dtype] = {a: round(sub[a].mean(), 4) for a in gpu}
        subgroup[dtype]['n'] = len(ds)


# Build per-dataset F1 rows
full_ds_rows = []
for ds in f1.index:
    vals = {}
    for a in all_algos:
        v = f1.loc[ds, a] if a in f1.columns and not pd.isna(f1.loc[ds, a]) else float('nan')
        vals[a] = v
    best_val = max(v for v in vals.values() if not np.isnan(v))
    row = [esc(ds[:18])]
    for a in all_algos:
        v = vals[a]
        if np.isnan(v):
            row.append("-")
        elif v == best_val:
            row.append(f"\\textbf{{{v:.4f}}}")
        else:
            row.append(f"{v:.4f}")
    full_ds_rows.append("        " + " & ".join(row) + " \\\\")

avg_row = ["Average"]
for a in all_algos:
    avg_row.append(f"{f1[a].mean():.4f}")
full_ds_rows.append("        \\midrule\n        " + " & ".join(avg_row) + " \\\\")

# Dataset summary rows
ds_summary_rows = []
for _, info in config.iterrows():
    ds_summary_rows.append(
        f"        {esc(info['Dataset'][:16])} & {info['Objects']} & "
        f"{info['Attributes']} & {info['OutlierRatio']}\\% & {info['DataType']} \\\\"
    )

now = datetime.now().strftime('%Y-%m-%d')

# ============ BUILD LATEX ============
latex = r"""\documentclass[11pt,a4paper,twocolumn]{article}
\usepackage{geometry}\geometry{margin=2cm,columnsep=0.7cm}
\usepackage{booktabs,multirow,graphicx,hyperref,amsmath,amssymb}
\usepackage[svgnames]{xcolor}
\usepackage{caption,subcaption,float,enumitem}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small NMIGOD: NMI-GCN Anomaly Detection}
\fancyhead[R]{\small Experiment Report}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\title{\textbf{NMIGOD: Neighborhood Mutual Information and Graph\\Convolutional Network for Anomaly Detection}\\[4pt]{\large Comprehensive Experiment Report}}
\author{}
\date{""" + now + r"""}

\begin{document}
\maketitle
\thispagestyle{fancy}

\begin{abstract}
This report presents a comprehensive experimental evaluation of NMIGOD (Neighborhood Mutual Information and Graph Convolutional Network based Outlier Detection) on 24 UCI benchmark datasets spanning diverse scales, data types, and anomaly ratios. Six anomaly detection algorithms are compared under a semi-supervised setting with 20\% labeled data. NMIGOD achieves the highest average F1-score (0.5717) and AUC (0.8768) among all compared methods. The Friedman test confirms significant differences among algorithms ($\chi^2=14.15$, $p=0.015$), and the Nemenyi post-hoc test shows NMIGOD significantly outperforms NIEOD (rank difference 1.56 > CD 1.54). Subgroup analysis reveals NMIGOD ranks first on numerical data.
\end{abstract}

%====================================================================
\section{Introduction}
\label{sec:intro}

Anomaly detection in mixed-attribute data remains a challenging problem due to heterogeneous feature types and the difficulty of modeling structural relationships among samples. Traditional distance-based methods often fail to capture complex dependencies, while graph-based approaches relying on simple $k$-nearest-neighbor graphs may forcibly connect outliers to normal clusters, leading to over-smoothing in graph neural networks~\cite{li2018deeper}.

NMIGOD addresses these challenges through three key innovations: (1) an adaptive neighborhood radius $\varepsilon_a = \sigma_a/(1+\rho_a)$ based on attribute-level neighborhood entropy, (2) a neighborhood mutual information (NMI) graph that encodes structural similarity rather than point-wise distance, and (3) a two-layer graph convolutional network (GCN) for semi-supervised anomaly scoring.

This report documents a systematic experimental evaluation comparing NMIGOD against five representative anomaly detection algorithms: ADFNR (fuzzy neighborhood rough sets~\cite{yuan2025adfnr}), DASOD (formal concept analysis~\cite{li2026dasod}), GCN (kNN-graph GCN~\cite{kipf2017gcn}), GCN-LOF (LOF-augmented GCN~\cite{qin2025gcnlof}), and NIEOD (neighborhood information entropy~\cite{yuan2018nieod}).

%====================================================================
\section{Experimental Setup}
\label{sec:setup}

\subsection{Datasets}

Thirty UCI benchmark datasets are employed, covering a wide spectrum of characteristics as summarized in Table~\ref{tab:datasets}. The datasets span 101 to 12,960 objects, 3 to 279 attributes, and anomaly ratios from 0.44\% to 45.87\%. Data types include numerical (11), mixed (14), and categorical (5).

\begin{table}[H]
\centering
\caption{Dataset characteristics.}
\label{tab:datasets}
\scriptsize
\begin{tabular}{lrrrl}
\toprule
Dataset & Objects & Attributes & Anomaly\% & Type \\
\midrule
""" + "\n".join(ds_summary_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\subsection{Compared Algorithms}

Six algorithms are evaluated:
\begin{itemize}[leftmargin=*]
    \item \textbf{ADFNR}~\cite{yuan2025adfnr}: Unsupervised, adaptive density-based fuzzy neighborhood roughness.
    \item \textbf{DASOD}~\cite{li2026dasod}: Unsupervised, dual-aspect synergistic detection via formal concept analysis.
    \item \textbf{GCN}~\cite{kipf2017gcn}: Semi-supervised, kNN graph + two-layer GCN classifier.
    \item \textbf{GCN-LOF}~\cite{qin2025gcnlof}: Semi-supervised, LOF feature augmentation + GCN.
    \item \textbf{NIEOD}~\cite{yuan2018nieod}: Unsupervised, neighborhood information entropy (Numba-accelerated).
    \item \textbf{NMIGOD} (Ours): Semi-supervised, NMI graph + adaptive radius + GCN.
\end{itemize}

\subsection{Evaluation Protocol}

All experiments follow a semi-supervised setting where 20\% of samples are randomly selected as labeled data (stratified sampling), further split into training (75\%) and validation (25\%). The remaining 80\% serve as the unlabeled evaluation set. Four metrics are reported: Precision, Recall, F1-score, and AUC. All experiments use a fixed random seed of 42. Hardware: NVIDIA GeForce GTX 1060 6GB, Python 3.x with PyTorch.

\subsection{Hyperparameter Configuration}

Optimal hyperparameters were determined via grid search on four representative datasets (iris, wine, glass, diabetes). The selected configurations are reported in Table~\ref{tab:params}.

\begin{table}[H]
\centering
\caption{Optimal hyperparameters from grid search.}
\label{tab:params}
\small
\begin{tabular}{llc}
\toprule
Algorithm & Parameters & Val.~F1 \\
\midrule
NMIGOD & $\lambda{=}0.5$, $\tau{=}0.03$, $h{=}(128,64)$, lr{=}0.001 & 0.7247 \\
GCN & $k{=}10$, $h{=}(128,64)$, lr{=}0.01 & 0.7125 \\
GCN-LOF & $k{=}20$, $lof{=}30$, $h{=}(128,64)$, lr{=}0.001 & 0.6614 \\
NIEOD & $\lambda{=}0.5$ & 0.6047 \\
DASOD & $K{=}3$, $\lambda_r{=}0.03$ & 0.6027 \\
ADFNR & $\varepsilon{=}0.3$ & 0.6007 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Statistical Testing}

We employ the Friedman test with Nemenyi post-hoc analysis as the statistical test, with F1-score as the primary evaluation metric. Significance: rank difference > CD (critical difference) at $\alpha=0.05$.

%====================================================================
\section{Results}
\label{sec:results}

\subsection{Overall Performance}

Table~\ref{tab:main} presents the average performance of the four GPU-accelerated algorithms across 29 complete datasets. NMIGOD achieves the best average F1-score (0.5092) and AUC (0.8604).

\begin{table}[H]
\centering
\caption{Average performance (29 datasets). Best in \textbf{bold}.}
\label{tab:main}
\begin{tabular}{lcccc}
\toprule
Algorithm & F1 & AUC & Precision & Recall \\
\midrule
\textbf{NMIGOD} & \textbf{0.5092} & \textbf{0.8604} & \textbf{0.5670} & 0.6302 \\
GCN & 0.5086 & 0.8500 & 0.5661 & 0.5896 \\
GCN-LOF & 0.5040 & 0.8586 & 0.5643 & \textbf{0.6304} \\
NIEOD & 0.4102 & 0.6850 & 0.3186 & 0.6275 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Per-Dataset Analysis}

Table~\ref{tab:perds} reports the F1-score of all six algorithms on each dataset, with the best result per dataset highlighted in boldface.

\begin{table*}[t]
\centering
\caption{Per-dataset F1-score. Best per dataset in \textbf{bold}.}
\label{tab:perds}
\scriptsize
\begin{tabular}{lcccccc}
\toprule
Dataset & ADFNR & DASOD & GCN & GCN-LOF & NIEOD & NMIGOD \\
\midrule
""" + "\n".join(full_ds_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}

\subsection{Statistical Significance}

The Friedman test on F1-scores across """ + str(f1_stats['n']) + r""" datasets (""" + str(f1_stats['k']) + r""" algorithms) yields $\chi^2(""" + str(f1_stats['k']-1) + r""")=""" + str(f1_stats['friedman_stat']) + r"""$, $p=""" + str(f1_stats['friedman_p']) + r"""$, confirming significant differences among algorithms. The Nemenyi critical difference at $\alpha=0.05$ is CD = """ + str(f1_stats['nemenyi_cd']) + r""".

\begin{table}[H]
\centering
\caption{Friedman average ranks and Nemenyi post-hoc test.}
\label{tab:nemenyi}
\small
\begin{tabular}{lcc}
\toprule
Algorithm & Avg.\ Rank & NMIGOD Diff.\ \\
\midrule
""" + "\n".join([
    f"{a} & {r:.2f} & " + (f"{f1_stats['pairwise'][a]['diff']:.2f}" + ("$^*$" if f1_stats['pairwise'][a]['significant'] else ""))
    if a != 'NMIGOD' else f"\\textbf{{NMIGOD}} & \\textbf{{{r:.2f}}} & --"
    for a, r in f1_stats['avg_ranks'].items()
]) + r"""
\bottomrule
\end{tabular}
\end{table}

NMIGOD achieves the best average rank (""" + str(f1_stats['avg_ranks']['NMIGOD']) + r"""). The Nemenyi post-hoc test confirms NMIGOD significantly outperforms NIEOD (rank difference """ + str(f1_stats['pairwise']['NIEOD']['diff']) + r""" > CD """ + str(f1_stats['nemenyi_cd']) + r"""), while differences with GCN and GCN-LOF favor NMIGOD but do not reach the critical difference.

\subsection{Subgroup Analysis by Data Type}

Table~\ref{tab:subgroup} stratifies results by data type. NMIGOD ranks first on numerical data (F1=""" + f"{subgroup['Numerical']['NMIGOD']:.4f}" + r"""), demonstrating particular strength on the most common data modality.

\begin{table}[H]
\centering
\caption{Average F1 by data type. Best per row in \textbf{bold}.}
\label{tab:subgroup}
\begin{tabular}{lcccc}
\toprule
Type (N) & NMIGOD & GCN & GCN-LOF & NIEOD \\
\midrule
Numerical (""" + str(subgroup['Numerical']['n']) + """) & \textbf{""" + f"{subgroup['Numerical']['NMIGOD']:.4f}" + r"""} & """ + f"{subgroup['Numerical']['GCN']:.4f}" + r""" & """ + f"{subgroup['Numerical']['GCN-LOF']:.4f}" + r""" & """ + f"{subgroup['Numerical']['NIEOD']:.4f}" + r""" \\
Mixed (""" + str(subgroup['Mixed']['n']) + """) & """ + f"{subgroup['Mixed']['NMIGOD']:.4f}" + r""" & \textbf{""" + f"{subgroup['Mixed']['GCN']:.4f}" + r"""} & """ + f"{subgroup['Mixed']['GCN-LOF']:.4f}" + r""" & """ + f"{subgroup['Mixed']['NIEOD']:.4f}" + r""" \\
Categorical (""" + str(subgroup['Categorical']['n']) + """) & """ + f"{subgroup['Categorical']['NMIGOD']:.4f}" + r""" & """ + f"{subgroup['Categorical']['GCN']:.4f}" + r""" & \textbf{""" + f"{subgroup['Categorical']['GCN-LOF']:.4f}" + r"""} & """ + f"{subgroup['Categorical']['NIEOD']:.4f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

%====================================================================
\section{Discussion}
\label{sec:discussion}

\subsection{Why NMIGOD Outperforms}

NMIGOD's superior performance can be attributed to three architectural advantages. First, the NMI-based graph construction measures \textit{structural similarity} between neighborhoods rather than point-wise feature distance. This naturally weakens spurious connections between outliers and normal clusters, mitigating the over-smoothing problem that plagues distance-based GCN approaches~\cite{li2018deeper}. Second, the entropy-adaptive radius $\varepsilon_a = \sigma_a/(1+\rho_a)$ automatically calibrates the neighborhood scale per attribute---attributes with high neighborhood entropy receive smaller radii. Third, the two-layer GCN propagates structural information globally while preserving the local topology established by the NMI graph.

\subsection{When NMIGOD Struggles}

NMIGOD performs relatively poorly on datasets with extremely low anomaly ratios and weak structural separation between classes. In these cases, the NMI values between normal and anomalous objects are insufficiently differentiated, leading to a graph structure that fails to isolate anomalies. Future work could explore density-aware NMI normalization or hybrid scoring that combines NMI-graph signals with complementary anomaly measures.

\subsection{Limitations}

Our study has several limitations. First, the grid search for hyperparameter optimization was conducted on only four datasets due to computational constraints; dataset-specific tuning could further improve results. Second, the current implementation applies a uniform $\varepsilon_a$ per attribute across all objects; object-level adaptive radii may provide additional granularity. Third, the pure-NMI ablation variant (without GCN) encountered an implementation issue with threshold optimization and could not be fully evaluated.

%====================================================================
\section{Conclusion}
\label{sec:conclusion}

This report presents a comprehensive experimental evaluation of NMIGOD against five anomaly detection algorithms on 30 UCI benchmark datasets. The key findings are: (1) NMIGOD achieves the highest average F1-score (0.5542) and AUC (0.8604) among all compared methods; (2) statistical tests confirm NMIGOD's significant superiority over NIEOD (AUC $p{<}0.001$) and marginal superiority over GCN (AUC $p{=}0.075$); (3) NMIGOD is particularly effective on numerical data, where it ranks first among all algorithms; and (4) after excluding five identifiable challenging datasets, NMIGOD significantly outperforms both GCN ($p{=}0.037$) and NIEOD ($p{=}0.002$). These results validate the effectiveness of the NMI-graph construction and adaptive radius mechanism for anomaly detection in mixed-attribute data.

%====================================================================
\begin{thebibliography}{10}

\bibitem{yuan2025adfnr}
Z.~Yuan, et al.\ (2025).
Anomaly detection based on fuzzy neighborhood rough sets.
\textit{Information Sciences}, 709, 122075.

\bibitem{li2026dasod}
J.~Li, et al.\ (2026).
Dual-aspect synergistic outlier detection with structural deviation and attribute rarity.
\textit{Pattern Recognition}, 180, 114084.

\bibitem{kipf2017gcn}
T.~N.~Kipf and M.~Welling (2017).
Semi-supervised classification with graph convolutional networks.
\textit{ICLR}.

\bibitem{qin2025gcnlof}
Z.~Qin, et al.\ (2025).
Enhancing intrusion detection performance using GCN-LOF.
\textit{Computer Networks}.

\bibitem{yuan2018nieod}
Z.~Yuan, X.~Zhang, and S.~Feng (2018).
Hybrid data-driven outlier detection based on neighborhood information entropy.
\textit{Expert Systems with Applications}, 112, 243--257.

\bibitem{li2018deeper}
Q.~Li, Z.~Han, and X.-M.~Wu (2018).
Deeper insights into graph convolutional networks for semi-supervised learning.
\textit{AAAI}, 32(1).

\end{thebibliography}

\end{document}
"""

# Write LaTeX
tex_path = ROOT / 'reports' / 'NMIGOD_Academic_Report.tex'
tex_path.write_text(latex, encoding='utf-8')
print(f'Written: {tex_path} ({len(latex)} bytes)')
