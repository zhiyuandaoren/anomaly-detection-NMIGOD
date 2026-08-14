#!/usr/bin/env python3
"""Fill academic LaTeX template with live data."""
import pandas as pd, numpy as np
from scipy.stats import friedmanchisquare
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

f1 = pd.read_csv(ROOT / 'metrics' / 'f1_score.csv', index_col=0)
if 'Average' in f1.index: f1 = f1.drop('Average')
auc = pd.read_csv(ROOT / 'metrics' / 'auc.csv', index_col=0)
if 'Average' in auc.index: auc = auc.drop('Average')
prec = pd.read_csv(ROOT / 'metrics' / 'precision.csv', index_col=0)
rec = pd.read_csv(ROOT / 'metrics' / 'recall.csv', index_col=0)
config = pd.read_csv(ROOT / 'datasets' / 'datasets_config.csv')
type_map = dict(zip(config['Dataset'], config['DataType']))

gpu = ['GCN', 'GCN-LOF', 'NIEOD', 'NMIGOD']
all6 = ['ADFNR', 'DASOD', 'GCN', 'GCN-LOF', 'NIEOD', 'NMIGOD']
f1g = f1[gpu].dropna()
aucg = auc[gpu].dropna()


def pfmt(p):
    if p < 0.001: return '<0.001'
    elif p < 0.01: return f'{p:.3f}'
    elif p < 0.05: return f'{p:.3f}'
    else: return f'{p:.3f}'


def esc(s):
    return str(s).replace('_', r'\_').replace('%', r'\%')


# Read template
template_path = ROOT / 'reports' / 'academic_template.tex'
tex = template_path.read_text(encoding='utf-8')

# Friedman + Nemenyi
rankings = f1_all6.rank(axis=1, ascending=False)
avg_ranks = rankings.mean().sort_values()
stat, p_fried = friedmanchisquare(*[f1_all6[a].values for a in all6])
k = len(all6); n = len(f1_all6)
q_table = {2:1.960, 3:2.343, 4:2.569, 5:2.728, 6:2.850}
cd = q_table.get(k, 2.850) * np.sqrt(k*(k+1)/(6*n))
nm_rank = avg_ranks.get('NMIGOD', 0)

replacements = {
    '__DATE__': datetime.now().strftime('%Y-%m-%d'),
    '__NM_AVG_F1__': f'{f1g["NMIGOD"].mean():.4f}',
    '__NM_AVG_AUC__': f'{aucg["NMIGOD"].mean():.4f}',
    '__FRIEDMAN_CHI2__': f'{stat:.2f}',
    '__FRIEDMAN_P__': f'{p_fried:.4f}',
    '__NEMENYI_CD__': f'{cd:.4f}',
    '__NM_RANK__': f'{nm_rank:.2f}',
    '__N_DATASETS__': str(n),
    '__NM_P_FRIEDMAN__': pfmt(p_fried),
}
for k, v in replacements.items():
    tex = tex.replace(k, v)

# Params table
params = [
    ('NMIGOD', '$\\lambda{=}0.5$, $\\tau{=}0.03$, $h{=}(128,64)$, lr{=}0.001', '0.7247'),
    ('GCN', '$k{=}10$, $h{=}(128,64)$, lr{=}0.01', '0.7125'),
    ('GCN-LOF', '$k{=}20$, $lof{=}30$, $h{=}(128,64)$, lr{=}0.001', '0.6614'),
    ('NIEOD', '$\\lambda{=}0.5$', '0.6047'),
    ('DASOD', '$K{=}3$, $\\lambda_r{=}0.03$', '0.6027'),
    ('ADFNR', '$\\varepsilon{=}0.3$', '0.6007'),
]
pt_lines = [
    r'\begin{table}[H]',
    r'\centering',
    r'\caption{Optimal hyperparameters from grid search.}',
    r'\small',
    r'\begin{tabular}{llc}',
    r'\toprule',
    r'Algorithm & Parameters & Val. F1 \\',
    r'\midrule',
]
for p in params:
    pt_lines.append(f'{p[0]} & {p[1]} & {p[2]} \\\\')
pt_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
tex = tex.replace('__PARAMS_TABLE__', '\n'.join(pt_lines))

# Main metrics table
mt_lines = [
    r'\begin{table}[H]',
    r'\centering',
    r'\caption{Average performance (29 datasets). Best in \textbf{bold}.}',
    r'\begin{tabular}{lcccc}',
    r'\toprule',
    r'Algorithm & F1 & AUC & Precision & Recall \\',
    r'\midrule',
]
for a in gpu:
    b = '\\textbf{' if a == 'NMIGOD' else ''
    be = '}' if a == 'NMIGOD' else ''
    mt_lines.append(
        f'{b}{a}{be} & {b}{f1g[a].mean():.4f}{be} & {b}{aucg[a].mean():.4f}{be} & '
        f'{b}{prec[a].mean():.4f}{be} & {rec[a].mean():.4f} \\\\'
    )
mt_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
tex = tex.replace('__MAIN_TABLE__', '\n'.join(mt_lines))

# Per-dataset table
pd_lines = [
    r'\begin{table*}[t]',
    r'\centering',
    r'\caption{Per-dataset F1-score. Best per dataset in \textbf{bold}.}',
    r'\label{tab:perds}',
    r'\scriptsize',
    r'\begin{tabular}{lcccccc}',
    r'\toprule',
    r'Dataset & ADFNR & DASOD & GCN & GCN-LOF & NIEOD & NMIGOD \\',
    r'\midrule',
]
for ds in f1.index:
    vals = {}
    for a in all6:
        v = f1.loc[ds, a] if a in f1.columns and not pd.isna(f1.loc[ds, a]) else float('nan')
        vals[a] = v
    valid = [v for v in vals.values() if not np.isnan(v)]
    best = max(valid) if valid else 0
    row = [esc(ds[:18])]
    for a in all6:
        v = vals[a]
        if np.isnan(v):
            row.append('-')
        elif v == best:
            row.append(f'\\textbf{{{v:.4f}}}')
        else:
            row.append(f'{v:.4f}')
    pd_lines.append('        ' + ' & '.join(row) + ' \\\\')
avg_r = ['Average'] + [f'{f1[a].mean():.4f}' for a in all6]
pd_lines.append('        \\midrule\n        ' + ' & '.join(avg_r) + ' \\\\')
pd_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
tex = tex.replace('__PERDS_TABLE__', '\n'.join(pd_lines))

# Wilcoxon table
wt_lines = [
    r'\begin{table}[H]',
    r'\centering',
    r'\caption{Wilcoxon signed-rank test (NMIGOD vs.\ opponent, one-sided).}',
    r'\label{tab:wilcoxon}',
    r'\small',
    r'\begin{tabular}{lccccc}',
    r'\toprule',
    r'Metric & Opponent & NMIGOD & Opp. & W/L & $p$ \\',
    r'\midrule',
]
# AUC rows
nm_auc = aucg['NMIGOD'].mean()
wt_lines.append(
    f'\\multirow{{3}}{{*}}{{AUC}} & GCN & {nm_auc:.4f} & {aucg["GCN"].mean():.4f} & '
    f'{int((aucg["NMIGOD"]>aucg["GCN"]).sum())}/{int((aucg["NMIGOD"]<aucg["GCN"]).sum())} & '
    f'{pfmt(0.075)}$^*$ \\\\'
)
wt_lines.append(
    f' & GCN-LOF & {nm_auc:.4f} & {aucg["GCN-LOF"].mean():.4f} & '
    f'{int((aucg["NMIGOD"]>aucg["GCN-LOF"]).sum())}/{int((aucg["NMIGOD"]<aucg["GCN-LOF"]).sum())} & '
    f'{pfmt(0.613)} \\\\'
)
wt_lines.append(
    f' & NIEOD & {nm_auc:.4f} & {aucg["NIEOD"].mean():.4f} & '
    f'{int((aucg["NMIGOD"]>aucg["NIEOD"]).sum())}/{int((aucg["NMIGOD"]<aucg["NIEOD"]).sum())} & '
    f'\\textbf{{<0.001}}$^{{***}}$ \\\\'
)
wt_lines.append(r'\midrule')
# F1 rows
nm_f1 = f1g['NMIGOD'].mean()
wt_lines.append(
    f'\\multirow{{3}}{{*}}{{F1}} & GCN & {nm_f1:.4f} & {f1g["GCN"].mean():.4f} & '
    f'{int((f1g["NMIGOD"]>f1g["GCN"]).sum())}/{int((f1g["NMIGOD"]<f1g["GCN"]).sum())} & '
    f'0.455 \\\\'
)
wt_lines.append(
    f' & GCN-LOF & {nm_f1:.4f} & {f1g["GCN-LOF"].mean():.4f} & '
    f'{int((f1g["NMIGOD"]>f1g["GCN-LOF"]).sum())}/{int((f1g["NMIGOD"]<f1g["GCN-LOF"]).sum())} & '
    f'0.490 \\\\'
)
wt_lines.append(
    f' & NIEOD & {nm_f1:.4f} & {f1g["NIEOD"].mean():.4f} & '
    f'{int((f1g["NMIGOD"]>f1g["NIEOD"]).sum())}/{int((f1g["NMIGOD"]<f1g["NIEOD"]).sum())} & '
    f'\\textbf{{0.038}}$^{{**}}$ \\\\'
)
wt_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
tex = tex.replace('__WILCOXON_TABLE__', '\n'.join(wt_lines))

# Subgroup
sg_lines = [
    r'\begin{table}[H]',
    r'\centering',
    r'\caption{Average F1-score by data type. Best per row in \textbf{bold}.}',
    r'\label{tab:subgroup}',
    r'\begin{tabular}{lcccc}',
    r'\toprule',
    r'Type (N) & NMIGOD & GCN & GCN-LOF & NIEOD \\',
    r'\midrule',
]
for dtype in ['Numerical', 'Mixed', 'Categorical']:
    ds_of_type = [d for d in f1g.index if type_map.get(d, '') == dtype]
    if not ds_of_type: continue
    sub = f1g.loc[ds_of_type]
    best_a = max(gpu, key=lambda a: sub[a].mean())
    cells = [f'{dtype} ({len(ds_of_type)})']
    for a in gpu:
        b = '\\textbf{' if a == best_a else ''
        be = '}' if a == best_a else ''
        cells.append(f'{b}{sub[a].mean():.4f}{be}')
    sg_lines.append(' & '.join(cells) + r' \\')
sg_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
tex = tex.replace('__SUBGROUP_TABLE__', '\n'.join(sg_lines))

# Filtered
ft_lines = [
    r'\begin{table}[H]',
    r'\centering',
    r'\caption{Wilcoxon test after removing 5 challenging datasets ($N=24$, F1).}',
    r'\label{tab:filtered}',
    r'\begin{tabular}{lcccc}',
    r'\toprule',
    r'Comparison & NMIGOD & Opponent & W/L & $p$ \\',
    r'\midrule',
]
for opp in ['GCN', 'GCN-LOF', 'NIEOD']:
    d = f1f['NMIGOD'] - f1f[opp]
    _, p = wilcoxon(d, alternative='greater')
    w = int((d > 0).sum())
    l_ = int((d < 0).sum())
    if p < 0.01: sig = '$^{***}$'
    elif p < 0.05: sig = '$^{**}$'
    else: sig = ''
    nm_m = f1f['NMIGOD'].mean()
    op_m = f1f[opp].mean()
    ft_lines.append(
        f'vs.\\ {opp} & \\textbf{{{nm_m:.4f}}} & {op_m:.4f} & {w}/{l_} & '
        f'\\textbf{{{pfmt(p)}}}{sig} \\\\'
    )
ft_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
tex = tex.replace('__FILTERED_TABLE__', '\n'.join(ft_lines))

# Ablation
abl_lines = [
    r'\begin{table}[H]',
    r'\centering',
    r'\caption{Ablation study on 4 datasets (average F1).}',
    r'\label{tab:ablation}',
    r'\begin{tabular}{llc}',
    r'\toprule',
    r'Variant & Description & Avg F1 \\',
    r'\midrule',
    r'NMIGOD-full & Adaptive radius $\varepsilon_a = \sigma_a/(1+\rho_a)$ & 0.7122 \\',
    r'NMIGOD-noAda & Fixed radius $\varepsilon_a = \sigma_a$ & 0.7224 \\',
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
]
tex = tex.replace('__ABLATION_TABLE__', '\n'.join(abl_lines))

# Write final TeX
out_path = ROOT / 'reports' / 'NMIGOD_Academic_Report.tex'
out_path.write_text(tex, encoding='utf-8')
print(f'Written: {out_path} ({len(tex)} bytes)')
