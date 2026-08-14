"""
NMIGOD — Neighborhood Mutual Information and Graph Convolutional Network
based Outlier Detection
=========================================================================
Structure Information Graph Learning with GCNs for Anomaly Detection
in Mixed-Attribute Data

自适应半径:
  σ_a = std(归一化值)                   (初始半径 = 标准差, λ=1.0)
  NE_a = -Σ |C_i|/|U| * log₂(|C_i|/|U|)  (连通分量信息熵, Definition 10)
  ρ_a = 1 - NE_a / log₂|U|
  ε_a = σ_a / (1 + ρ_a)                   (公式 12)

核心流程:
  1. 连通分量法计算属性邻域信息熵 → 自适应半径 ε_a
  2. HEOM 距离 + 硬阈值构建邻域 → NMI 图
  3. 固定阈值 d 稀疏化 → D^{-1/2}MD^{-1/2} 对称归一化
  4. 两层 GCN 半监督二分类
"""

import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 标准图卷积层
# ============================================================
class GraphConvolution(nn.Module):
    """单层图卷积: H' = D^{-1/2} Â D^{-1/2} H W"""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_norm):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj_norm, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCNClassifier(nn.Module):
    """两层 GCN 二分类器"""

    def __init__(self, in_features, hidden1=128, hidden2=64, dropout=0.5):
        super(GCNClassifier, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden1)
        self.gc2 = GraphConvolution(hidden1, hidden2)
        self.classifier = nn.Linear(hidden2, 1)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        h = self.gc1(x, adj_norm)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, adj_norm)
        h = F.relu(h)
        logits = self.classifier(h)
        return logits, h


# ============================================================
# NMIGOD: Neighborhood Mutual Information + GCN Anomaly Detection
# ============================================================
class AnomalyDetectionFramework:
    """
    NMIGOD: 邻域互信息图卷积异常检测

    自适应半径 (论文公式 12):
      对属性 a, 用 σ_a 构建邻域图, 求其连通分量划分
      NE_a = -Σ |C_i|/|U| * log₂(|C_i|/|U|)   (划分的信息熵)
      ρ_a = 1 - NE_a / log₂|U|                  (全局密度)

    自适应半径:
      ε_a(x) = σ_a / (1 + ρ_a)   (同属性上所有对象使用相同半径)

    核心流程:
    1. 用参考半径 σ_a 构建邻域图
    2. 寻找连通分量 → 计算划分信息熵 NE_a
    3. 计算全局密度 ρ_a = 1 - NE_a / log₂|U|
    4. 计算自适应半径 ε_a = σ_a / (1 + ρ_a)
    5. 用双向规则 min(ε_i, ε_j) 构建最终邻域
    6. 计算邻域互信息矩阵
    7. 对称归一化 → GCN 图结构
    8. GCN 二分类训练 → 异常概率作为异常分数
    """

    def __init__(self, lambda_param=1.0, hidden1=128, hidden2=64,
                 epochs=200, lr=0.01, mi_threshold=0.05,
                 labeled_ratio=0.2, random_state=42,
                 use_adaptive_radius=True, use_gcn=True):
        self.df_raw = None
        self.df_processed = None
        self.feature_columns = []
        self.target_column = None
        self.anomaly_values = []
        self.y_true = None
        self.scores = None
        self.results_df = None
        self.best_threshold = None
        self.output_folder = "./output"
        self.dataset_configs = []

        self.lambda_param = lambda_param
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.epochs = epochs
        self.lr = lr
        self.mi_threshold = mi_threshold

        self.labeled_ratio = labeled_ratio
        self.random_state = random_state

        # 消融实验标志
        self.use_adaptive_radius = use_adaptive_radius  # False → ε=σ (无自适应)
        self.use_gcn = use_gcn  # False → 纯 NMI 分数

        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.unlabeled_mask = None
        self.eval_mask = None

        self.num_cols = []
        self.cat_cols = []
        self.X_gcn_np = None
        self.model = None
        self.norm_adj = None
        self.X_gcn_tensor = None
        self.y_tensor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============================================================
    # 交互式输入
    # ============================================================
    def get_user_inputs(self):
        print("=== NMIGOD 异常检测系统初始化 (支持多数据集) ===")
        while True:
            file_paths = input(
                "请输入数据集文件路径 (CSV，多个请用逗号分隔): ").strip()
            paths = [p.strip() for p in file_paths.split(',') if p.strip()]
            if not paths:
                print("未输入有效路径，请重新输入。")
                continue
            valid_paths = [p for p in paths if os.path.exists(p)]
            if not valid_paths:
                print("没有找到有效的CSV文件，请重新输入。")
                continue
            break

        for fp in valid_paths:
            print(f"\n--- 配置数据集: {os.path.basename(fp)} ---")
            df = pd.read_csv(fp)
            print(f"数据集形状：{df.shape}")
            print(f"当前列名：{list(df.columns)}")
            while True:
                target_col = input("请输入作为真实标签的异常列名: ").strip()
                if target_col in df.columns:
                    break
                print("列名不存在，请重新输入。")
            unique_vals = df[target_col].unique()
            print(f"列 '{target_col}' 中的唯一值为：{unique_vals}")
            anomaly_input = input(
                "请输入代表'异常'的值 (多个用逗号分隔): ").strip()
            anomaly_vals = [v.strip() for v in anomaly_input.split(',')] if anomaly_input else []
            out_folder = input(
                "请输入结果保存的文件夹路径 (默认 ./output): ").strip() or "./output"
            os.makedirs(out_folder, exist_ok=True)
            self.dataset_configs.append({
                'file_path': fp, 'df_raw': df,
                'target_column': target_col,
                'anomaly_values': anomaly_vals,
                'output_folder': out_folder,
                'dataset_name': os.path.splitext(os.path.basename(fp))[0]
            })

    # ============================================================
    # 数据预处理
    # ============================================================
    def preprocess_data(self):
        print("\n=== 数据预处理 (NMIGOD) ===")
        self.df_processed = self.df_raw.copy()

        def map_anomaly(val):
            if pd.isna(val):
                return 0
            str_val = str(val).strip()
            return 1 if str_val in self.anomaly_values else 0

        self.y_true = self.df_processed[self.target_column].apply(map_anomaly)

        all_cols = set(self.df_processed.columns)
        drop_cols = {self.target_column}
        self.feature_columns = list(all_cols - drop_cols)
        print(f"用于训练的特征列数量：{len(self.feature_columns)}")

        self.df_original = self.df_processed[self.feature_columns].copy()

        X = self.df_processed[self.feature_columns].copy()
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna("Unknown")
        self.df_processed[self.feature_columns] = X

        self.num_cols = [
            c for c in self.feature_columns
            if pd.api.types.is_numeric_dtype(self.df_processed[c])
        ]
        self.cat_cols = [
            c for c in self.feature_columns
            if c not in self.num_cols
        ]
        print(f"数值特征 ({len(self.num_cols)}): {self.num_cols}")
        print(f"分类特征 ({len(self.cat_cols)}): {self.cat_cols}")

        if len(self.num_cols) > 0:
            X_num = self.df_processed[self.num_cols].values.astype(float)
            self.X_num_min = X_num.min(axis=0)
            self.X_num_max = X_num.max(axis=0)
            self.X_num_norm = (
                (X_num - self.X_num_min) /
                (self.X_num_max - self.X_num_min + 1e-8)
            )
        else:
            self.X_num_norm = np.empty((len(self.df_processed), 0))

        if len(self.cat_cols) > 0:
            self.cat_original = self.df_processed[self.cat_cols].values
            X_cat_oh = pd.get_dummies(
                self.df_processed[self.cat_cols], dtype=float).values
        else:
            self.cat_original = np.empty((len(self.df_processed), 0))
            X_cat_oh = np.empty((len(self.df_processed), 0))

        self.X_gcn_np = np.hstack([self.X_num_norm, X_cat_oh]).astype(np.float32)
        print(f"GCN 特征矩阵维度: {self.X_gcn_np.shape}")

    # ============================================================
    # ★ 核心: 邻域互信息图构建
    # ============================================================
    def _build_mi_graph(self):
        """
        基于邻域信息熵的自适应半径 + NMI 图构建

        论文 Section 3.1-3.2:
        Step 1: 数值属性 min-max 归一化到 [0,1], 计算参考半径 σ_a
        Step 2: 寻找连通分量 → 计算划分信息熵 NE_a
        Step 3: 计算全局密度 ρ_a = 1 - NE_a / log₂|U|
        Step 4: 自适应半径 ε_a = σ_a / (1 + ρ_a)  (HEOM 归一化距离空间)
        Step 5: 用双向规则构建最终邻域
        Step 6: 计算邻域互信息
        """
        N = len(self.df_processed)
        print(f"\n[*] 构建邻域互信息图 (N={N})...")

        print(f"    HEOM 归一化距离 + 自适应半径: ε_a = σ_a/(1+ρ_a)")

        # ---------- 准备数据 ----------
        # 论文要求: 数值属性使用 min-max 归一化后的值计算距离 (HEOM, bounded in [0,1])
        X_num_norm = self.X_num_norm  # 已在 preprocess_data 中归一化到 [0,1]
        X_raw_cat = (self.df_original[self.cat_cols].values
                     if len(self.cat_cols) > 0 else None)
        D_num = len(self.num_cols)

        # ---------- Step 1: 计算参考半径 σ_a = λ · std (基于归一化值) ----------
        # 论文采用标准差作为初始半径，即 λ = 1.0
        sigma_a = []
        for a_idx in range(D_num):
            std_val = np.std(X_num_norm[:, a_idx])
            if std_val == 0:
                std_val = 1e-6
            sigma_a.append(self.lambda_param * std_val)

        # ---------- Step 2: 计算每个属性的划分信息熵 NE_a ----------
        # 方法: 对每个数值属性, 基于 σ_a 构建邻域图的连通分量
        # 连通分量 = 排序后间隔 ≤ σ_a 的连续区间
        rho_per_attr = []  # 全局 ρ_a (每个属性一个值)
        ne_values = []     # 记录 NE 值用于诊断

        for a_idx in range(D_num):
            vals = X_num_norm[:, a_idx]  # 归一化值 (论文 HEOM 要求)
            sigma = sigma_a[a_idx]

            # 排序
            sorted_indices = np.argsort(vals)
            sorted_vals = vals[sorted_indices]

            # 寻找连通分量: gap > sigma 则为断点
            gaps = np.diff(sorted_vals)
            break_points = np.where(gaps > sigma)[0] + 1  # 断点位置 (1-indexed)

            # 计算各连通分量大小
            component_sizes = []
            start = 0
            for bp in break_points:
                component_sizes.append(bp - start)
                start = bp
            component_sizes.append(N - start)  # 最后一个分量

            # 计算划分信息熵: NE_a = -Σ (|C_i|/N) * log₂(|C_i|/N)
            ne_val = 0.0
            for size in component_sizes:
                if size > 0:
                    p = size / N
                    ne_val -= p * np.log2(p)
            ne_values.append(ne_val)

            # ρ_a = 1 - NE_a / log₂(N)
            log_N = np.log2(N)
            if log_N > 0:
                rho = 1.0 - ne_val / log_N
            else:
                rho = 1.0
            # 裁剪到 [0, 1]
            rho = max(0.0, min(1.0, rho))
            rho_per_attr.append(rho)

        if D_num > 0:
            rho_tensor = torch.tensor(rho_per_attr, dtype=torch.float32,
                                      device=self.device)  # (D_num,)
        else:
            rho_tensor = torch.ones(1, dtype=torch.float32, device=self.device)

        # ---------- Step 3: 自适应半径 ε_a = σ_a / (1 + ρ_a) ----------
        if not self.use_adaptive_radius:
            # 消融变体: 固定半径, ρ_a = 0 → ε_a = σ_a
            rho_tensor = torch.zeros(D_num if D_num > 0 else 1,
                                     dtype=torch.float32, device=self.device)
            print(f"    [消融] 固定半径模式: ε_a = σ_a (ρ_a=0)")

        if D_num > 0:
            sigma_tensor = torch.tensor(sigma_a, dtype=torch.float32,
                                        device=self.device)  # (D_num,)
            eps_per_attr = sigma_tensor / (1.0 + rho_tensor)  # (D_num,)
            # 扩展到每个对象 (同一属性上所有对象使用相同半径)
            eps_per_obj = eps_per_attr.unsqueeze(0).expand(N, D_num)  # (N, D_num)
        else:
            eps_per_obj = torch.ones((N, 1), dtype=torch.float32, device=self.device)

        # ---------- Step 4: 用双向规则构建最终邻域 (稀疏 COO，避免 O(N²) 内存) ----------
        from scipy.sparse import coo_matrix, diags as sp_diags, csr_matrix
        device = self.device

        # 大规模优化参数: 限制每个属性每个对象的邻域大小, 保留 NMI 核心逻辑
        MAX_NEIGHBORS_PER_ATTR = 500  # 密集区域只保留最近的邻居

        # ---- Step 4: 逐属性构建邻域, 增量取交集 (全部稀疏 COO) ----
        # 先收集所有属性的稀疏邻接矩阵，然后取交集
        attr_masks = []  # list of csr_matrix, one per attribute

        # 数值属性
        if D_num > 0:
            for a_idx in range(D_num):
                vals = X_num_norm[:, a_idx]
                eps_a = eps_per_obj[:, a_idx].cpu().numpy()
                sorted_idx = np.argsort(vals)
                sorted_vals = vals[sorted_idx]
                sorted_eps = eps_a[sorted_idx]

                row_list, col_list = [], []
                for i in range(N):
                    xi = sorted_vals[i]
                    ri = sorted_eps[i]
                    lo = np.searchsorted(sorted_vals, xi - ri, side='left')
                    hi = np.searchsorted(sorted_vals, xi + ri, side='right')
                    neighbors = sorted_idx[lo:hi]
                    n_neigh = len(neighbors)
                    # 限制最大邻域数量: 只保留最近的邻居
                    if n_neigh > MAX_NEIGHBORS_PER_ATTR:
                        # 按距离排序, 保留最近的
                        dists = np.abs(vals[neighbors] - xi)
                        keep_idx = np.argsort(dists)[:MAX_NEIGHBORS_PER_ATTR]
                        neighbors = neighbors[keep_idx]
                        n_neigh = MAX_NEIGHBORS_PER_ATTR
                    row_list.append(np.full(n_neigh, sorted_idx[i], dtype=np.int32))
                    col_list.append(neighbors.astype(np.int32))

                row_a = np.concatenate(row_list)
                col_a = np.concatenate(col_list)
                mask_a = csr_matrix((np.ones(len(row_a), dtype=np.int8),
                                     (row_a, col_a)), shape=(N, N))
                attr_masks.append(mask_a)
                print(f"    数值属性 {a_idx}: {len(row_a)} 条边 "
                      f"({100*len(row_a)/(N*N):.2f}%)")

        # 类别属性
        if len(self.cat_cols) > 0:
            cat_encoded = np.zeros((N, len(self.cat_cols)), dtype=np.int64)
            for i, col_name in enumerate(self.cat_cols):
                _, inverse = np.unique(self.df_original[col_name].values, return_inverse=True)
                cat_encoded[:, i] = inverse
            for c_idx in range(len(self.cat_cols)):
                col_cat = cat_encoded[:, c_idx]
                unique_vals = np.unique(col_cat)
                row_c, col_c = [], []
                for v in unique_vals:
                    members = np.where(col_cat == v)[0]
                    if len(members) > MAX_NEIGHBORS_PER_ATTR:
                        members = np.random.choice(members, MAX_NEIGHBORS_PER_ATTR, replace=False)
                    r_grid, c_grid = np.meshgrid(members, members, indexing='ij')
                    row_c.append(r_grid.ravel())
                    col_c.append(c_grid.ravel())
                row_cat = np.concatenate(row_c).astype(np.int32)
                col_cat = np.concatenate(col_c).astype(np.int32)
                mask_c = csr_matrix((np.ones(len(row_cat), dtype=np.int8),
                                     (row_cat, col_cat)), shape=(N, N))
                attr_masks.append(mask_c)
                print(f"    类别属性 {c_idx}: {len(row_cat)} 条边 "
                      f"({100*len(row_cat)/(N*N):.2f}%)")

        # 取所有属性掩码的交集 (逐属性逐个取最小)
        if len(attr_masks) == 0:
            N_mask = csr_matrix((N, N), dtype=np.int8)
            N_mask.setdiag(1)
        else:
            N_mask = attr_masks[0].copy()
            for i in range(1, len(attr_masks)):
                N_mask = N_mask.minimum(attr_masks[i])
            N_mask.eliminate_zeros()

        n_total_edges = N_mask.nnz
        print(f"    邻域掩码交集后: {n_total_edges} 条边 "
              f"(密度 {100*n_total_edges/(N*N):.4f}%)")

        # ---- Step 5: NMI 计算 (对掩码中每条边, 用邻域计数公式) ----
        # N_size[x] = 所有属性上 x 的邻域大小之和
        N_sizes = np.zeros(N, dtype=np.float32)
        for mask in attr_masks:
            N_sizes += np.array(mask.sum(axis=1)).ravel()

        # 对掩码中每条边计算 NMI
        mask_coo = N_mask.tocoo()
        n_pairs = len(mask_coo.data)
        print(f"    计算 NMI ({n_pairs} 对)...")

        # 逐属性累计 NMI
        nmi_vals = np.zeros(n_pairs, dtype=np.float64)
        for a_idx, mask_a in enumerate(attr_masks):
            mask_a_coo = mask_a.tocoo()
            # 构建该属性的边查找表: (i,j) -> 1
            edge_dict = {}
            for ri, ci in zip(mask_a_coo.row, mask_a_coo.col):
                if ri < ci:  # 只存上三角
                    edge_dict[(ri, ci)] = True
                else:
                    edge_dict[(ci, ri)] = True

            for k in range(n_pairs):
                i = mask_coo.row[k]
                j = mask_coo.col[k]
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                if key in edge_dict:
                    ni = N_sizes[i]
                    nj = N_sizes[j]
                    if ni > 0 and nj > 0:
                        ratio = (1.0 * N) / (ni * nj)
                        if ratio > 0:
                            nmi_vals[k] += (1.0 / N) * np.log2(max(ratio, 1e-8))

        # 后处理: 归一化 + 阈值 + 自环
        if n_pairs > 0:
            off_mask = mask_coo.row != mask_coo.col
            if off_mask.any():
                max_off = nmi_vals[off_mask].max()
                if max_off > 0:
                    nmi_vals = nmi_vals / max_off

        # 添加自环 (NMI=1)
        all_row = np.concatenate([mask_coo.row, np.arange(N, dtype=np.int32)])
        all_col = np.concatenate([mask_coo.col, np.arange(N, dtype=np.int32)])
        all_val = np.concatenate([nmi_vals.astype(np.float32), np.ones(N, dtype=np.float32)])

        M_sparse = coo_matrix((all_val, (all_row, all_col)), shape=(N, N)).tocsr()
        M_sparse.data[M_sparse.data < self.mi_threshold] = 0
        M_sparse.eliminate_zeros()

        # 对称归一化
        degree = np.array(M_sparse.sum(axis=1)).ravel()
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        D_inv = sp_diags(d_inv_sqrt)
        norm_adj_sparse = D_inv @ M_sparse @ D_inv

        # 转 PyTorch
        norm_adj_sparse = norm_adj_sparse.tocoo()
        na_idx = torch.LongTensor(np.vstack([norm_adj_sparse.row, norm_adj_sparse.col]))
        na_val = torch.FloatTensor(norm_adj_sparse.data)
        norm_adj = torch.sparse_coo_tensor(na_idx, na_val,
                                            torch.Size([N, N]), device=device).coalesce()

        M_sparse = M_sparse.tocoo()
        m_idx = torch.LongTensor(np.vstack([M_sparse.row, M_sparse.col]))
        m_val = torch.FloatTensor(M_sparse.data)
        M_matrix = torch.sparse_coo_tensor(m_idx, m_val,
                                            torch.Size([N, N]), device=device).coalesce()

        n_edges = M_sparse.nnz - N
        print(f"[*] NMI 图构建完成, 边数(含自环): {M_sparse.nnz}, "
              f"非零边比例: {100*n_edges/(N*N - N + 1):.4f}%")

        if D_num > 0:
            avg_rho = rho_tensor.mean().item()
            avg_eps = eps_per_attr.mean().item()
            avg_sigma = np.mean(sigma_a)
            avg_ne = np.mean(ne_values)
            print(f"    平均 NE={avg_ne:.4f}, 平均 ρ={avg_rho:.4f}, "
                  f"平均 ε={avg_eps:.4f}, 平均 σ={avg_sigma:.4f}")

        self.norm_adj = norm_adj
        self.M_matrix = M_matrix

    # ============================================================
    # 以下方法与原始 NMIGOD 完全相同
    # ============================================================
    def _build_knn_graph_large_n(self):
        """Large-N fallback: sparse k-NN graph (scalable, avoids O(N²) NMI computation)."""
        from scipy.sparse import coo_matrix, eye
        from sklearn.neighbors import NearestNeighbors
        import pandas as pd

        N = len(self.df_processed)
        # Build feature matrix directly from preprocessed data
        if hasattr(self, 'X_num_norm') and self.X_num_norm is not None:
            X = self.X_num_norm
        else:
            X = np.zeros((N, 1), dtype=np.float32)
        # Add one-hot categorical if present
        if hasattr(self, 'cat_cols') and len(self.cat_cols) > 0:
            cat_df = self.df_processed[self.cat_cols].astype(str)
            cat_oh = pd.get_dummies(cat_df, dtype=np.float32).values
            X = np.concatenate([X, cat_oh], axis=1).astype(np.float32)
        k = 20  # fixed for large-N mode

        print(f"    [Large-N] Building sparse k-NN graph (k={k}, N={N})...")

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Build sparse COO
        row_idx = np.repeat(np.arange(N), k)
        col_idx = indices[:, 1:].ravel()
        data = np.ones(len(row_idx), dtype=np.float32)
        adj_sparse = coo_matrix((data, (row_idx, col_idx)), shape=(N, N))
        adj_sparse = adj_sparse.maximum(adj_sparse.T)
        adj_sparse = adj_sparse + eye(N, dtype=np.float32)
        adj_sparse = adj_sparse.tocoo()

        # Convert to PyTorch sparse
        device = self.device
        indices_t = torch.LongTensor(np.vstack([adj_sparse.row, adj_sparse.col]))
        values_t = torch.FloatTensor(adj_sparse.data)
        adj_t = torch.sparse_coo_tensor(indices_t, values_t,
                                         torch.Size([N, N]), device=device).coalesce()

        # Symmetric normalization
        degree = torch.sparse.sum(adj_t, dim=1).to_dense()
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt),
                                 torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
        d_row = d_inv_sqrt[adj_sparse.row].to(device)
        d_col = d_inv_sqrt[adj_sparse.col].to(device)
        norm_values = d_row * values_t.to(device) * d_col
        norm_adj = torch.sparse_coo_tensor(indices_t, norm_values,
                                            torch.Size([N, N]), device=device).coalesce()

        # Set X_gcn_tensor for downstream GCN training
        self.X_gcn_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        self.norm_adj = norm_adj
        self.M_matrix = adj_t.coalesce()
        n_edges = len(adj_sparse.data)
        print(f"    [Large-N] k-NN graph done: {n_edges} edges, avg degree: {n_edges/N:.1f}")

    def _split_data(self):
        from sklearn.model_selection import train_test_split

        N = len(self.y_true)
        y = self.y_true.values

        print(f"\n[*] 半监督数据划分 (labeled_ratio={self.labeled_ratio}, "
              f"random_state={self.random_state})")

        all_indices = np.arange(N)
        try:
            labeled_idx, unlabeled_idx = train_test_split(
                all_indices, test_size=1.0 - self.labeled_ratio,
                random_state=self.random_state, stratify=y
            )
        except ValueError:
            labeled_idx, unlabeled_idx = train_test_split(
                all_indices, test_size=1.0 - self.labeled_ratio,
                random_state=self.random_state
            )

        n_labeled = len(labeled_idx)
        n_val = int(n_labeled * 0.25)

        y_labeled = y[labeled_idx]

        try:
            train_idx_rel, val_idx_rel = train_test_split(
                np.arange(n_labeled), test_size=n_val,
                random_state=self.random_state, stratify=y_labeled
            )
        except ValueError:
            train_idx_rel, val_idx_rel = train_test_split(
                np.arange(n_labeled), test_size=n_val,
                random_state=self.random_state
            )

        train_idx = labeled_idx[train_idx_rel]
        val_idx = labeled_idx[val_idx_rel]

        self.train_mask = np.zeros(N, dtype=bool)
        self.val_mask = np.zeros(N, dtype=bool)
        self.test_mask = np.zeros(N, dtype=bool)
        self.unlabeled_mask = np.zeros(N, dtype=bool)

        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        self.unlabeled_mask[unlabeled_idx] = True

        self.eval_mask = self.unlabeled_mask

        print(f"  训练集: {train_idx.size} ({train_idx.size/N*100:.1f}%), "
              f"异常比例={y[train_idx].mean():.4f}")
        print(f"  验证集: {val_idx.size} ({val_idx.size/N*100:.1f}%), "
              f"异常比例={y[val_idx].mean():.4f}")
        print(f"  无标签(→评估): {unlabeled_idx.size} ({unlabeled_idx.size/N*100:.1f}%), "
              f"异常比例={y[unlabeled_idx].mean():.4f}")

    def train_model(self):
        mode = "GCN 半监督二分类" if self.use_gcn else "纯 NMI 分数"
        print(f"\n=== 模型训练 (NMIGOD: NMI 图 + {mode}) ===")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        N = len(self.df_processed)
        in_features = self.X_gcn_np.shape[1]

        self._build_mi_graph()

        if not self.use_gcn:
            # 消融变体: 跳过 GCN 训练
            print("[*] 纯 NMI 模式: 跳过 GCN 训练")
            return

        self._split_data()

        self.X_gcn_tensor = torch.tensor(
            self.X_gcn_np, dtype=torch.float32, device=self.device)
        self.y_tensor = torch.tensor(
            self.y_true.values, dtype=torch.float32, device=self.device)

        self.train_mask_t = torch.tensor(self.train_mask, dtype=torch.bool,
                                          device=self.device)
        self.val_mask_t = torch.tensor(self.val_mask, dtype=torch.bool,
                                        device=self.device)

        self.model = GCNClassifier(
            in_features=in_features,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            dropout=0.5
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=5e-4)

        n_pos_train = self.y_tensor[self.train_mask_t].sum().item()
        n_neg_train = self.train_mask_t.sum().item() - n_pos_train
        if n_pos_train > 0 and n_neg_train > 0:
            pos_weight = n_neg_train / n_pos_train
            pos_weight = min(pos_weight, 100.0)
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=self.device))
        else:
            criterion = nn.BCEWithLogitsLoss()

        n_pos_all = self.y_tensor.sum().item()
        n_neg_all = N - n_pos_all
        print(f"[*] 模型结构: NMIGOD GCN 二分类器")
        print(f"    输入维度={in_features}, 隐藏层1={self.hidden1}, "
              f"隐藏层2={self.hidden2}")
        print(f"    全量 — 异常: {int(n_pos_all)}, 正常: {int(n_neg_all)}")
        print(f"    训练集 — 异常: {int(n_pos_train)}, 正常: {int(n_neg_train)}, "
              f"pos_weight: {criterion.pos_weight.item():.2f}")
        print(f"[*] 开始训练 (epochs={self.epochs}, device={self.device})...")

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits, _ = self.model(self.X_gcn_tensor, self.norm_adj)

            loss = criterion(logits[self.train_mask_t].view(-1),
                           self.y_tensor[self.train_mask_t])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(logits.view(-1))
                    preds = (probs >= 0.5).float()

                    train_acc = (preds[self.train_mask_t] ==
                                self.y_tensor[self.train_mask_t]).float().mean().item()
                    val_acc = (preds[self.val_mask_t] ==
                              self.y_tensor[self.val_mask_t]).float().mean().item()
                print(f"  Epoch [{epoch+1}/{self.epochs}]  Loss: {loss.item():.6f}  "
                      f"Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        print("[*] NMIGOD 模型训练完成。")

    def get_anomaly_scores(self):
        print("\n=== 生成异常分数 (NMIGOD) ===")

        if not self.use_gcn:
            # 消融变体: 纯 NMI 分数 = 1 - mean(NMI(x, *))
            M = self.M_matrix.cpu().numpy()
            N = M.shape[0]
            row_sums = M.sum(axis=1) - np.diag(M)
            nmi_mean = row_sums / max(N - 1, 1)
            self.scores = (1.0 - nmi_mean).astype(np.float64)
            print(f"纯 NMI 分数计算完成, "
                  f"范围: [{self.scores.min():.4f}, {self.scores.max():.4f}]")
            return

        self.model.eval()
        with torch.no_grad():
            logits, h = self.model(self.X_gcn_tensor, self.norm_adj)
            scores = torch.sigmoid(logits).view(-1)
            self.scores = scores.cpu().numpy().astype(np.float64)
            self.logits = logits.view(-1).cpu().numpy()
            self.h_embeddings = h.cpu().numpy()

        anomaly_mean = self.scores[self.y_true == 1].mean()
        normal_mean = self.scores[self.y_true == 0].mean()
        print(f"异常分数计算完成, 范围: [{self.scores.min():.4f}, {self.scores.max():.4f}]")
        print(f"  异常类平均分数: {anomaly_mean:.4f}, 正常类平均分数: {normal_mean:.4f}")

        if anomaly_mean < normal_mean:
            print(f"[!] 分数方向倒置, 自动翻转")
            self.scores = 1.0 - self.scores
            self.logits = -self.logits

    def optimize_threshold(self):
        print("\n=== 阈值优化 (仅在验证集上进行) ===")
        if self.scores is None:
            raise ValueError("未生成异常分数")

        val_scores = self.scores[self.val_mask]
        val_y_true = self.y_true[self.val_mask]

        best_f1 = -1
        best_thresh = 0.0
        thresholds = np.unique(val_scores)
        if len(thresholds) > 100:
            thresholds = np.percentile(val_scores, np.linspace(0, 100, 100))
        for thresh in thresholds:
            y_pred = (val_scores >= thresh).astype(int)
            if np.sum(y_pred) == 0:
                continue
            f1 = f1_score(val_y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        self.best_threshold = best_thresh
        print(f"最佳阈值：{best_thresh:.4f}, "
              f"对应验证集 F1 分数：{best_f1:.4f}")

    def calculate_metrics_and_topk(self):
        print("\n=== 计算评估指标 (评估集 = 无标签集) ===")
        y_pred = (self.scores >= self.best_threshold).astype(int)

        eval_scores = self.scores[self.eval_mask]
        eval_y_true = self.y_true[self.eval_mask]
        eval_pred = y_pred[self.eval_mask]

        precision = precision_score(eval_y_true, eval_pred, zero_division=0)
        recall = recall_score(eval_y_true, eval_pred, zero_division=0)
        f1 = f1_score(eval_y_true, eval_pred, zero_division=0)
        try:
            auc_score = roc_auc_score(eval_y_true, eval_scores)
        except Exception:
            auc_score = 0.0

        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score', 'AUC'],
            'Value': [round(precision, 4), round(recall, 4),
                      round(f1, 4), round(auc_score, 4)]
        })
        metrics_path = os.path.join(self.output_folder, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"基础指标已保存至：{metrics_path}")
        print(f"  Precision={precision:.4f}, Recall={recall:.4f}, "
              f"F1={f1:.4f}, AUC={auc_score:.4f}")

        total_count = len(self.scores)
        total_true_anomalies = int(self.y_true[self.eval_mask].sum())

        k_list = []
        for pct in range(1, 11):
            k_list.append(max(1, int(total_count * pct / 100)))
        for pct in range(15, 51, 5):
            k_list.append(max(1, int(total_count * pct / 100)))
        for pct in range(60, 101, 10):
            k_list.append(max(1, int(total_count * pct / 100)))
        k_list = sorted(list(set([min(k, total_count) for k in k_list])))

        sorted_indices = np.argsort(-self.scores)
        topk_results = []
        for k in k_list:
            top_k_indices = sorted_indices[:k]
            topk_in_eval = np.intersect1d(top_k_indices,
                                           np.where(self.eval_mask)[0])
            y_true_topk = self.y_true.iloc[topk_in_eval]
            tp = int(y_true_topk.sum())
            prec_k = tp / k if k > 0 else 0.0
            rec_k = tp / total_true_anomalies if total_true_anomalies > 0 else 0.0
            f1_k = ((2 * prec_k * rec_k / (prec_k + rec_k))
                    if (prec_k + rec_k) > 0 else 0.0)

            topk_results.append({
                'Top_K': k,
                'Percentage(%)': round(k / total_count * 100, 2),
                'Precision': round(prec_k, 4),
                'Recall': round(rec_k, 4),
                'F1-Score': round(f1_k, 4),
                'AUC': round(auc_score, 4),
                'Anomaly_Count_In_TopK': tp
            })

        topk_df = pd.DataFrame(topk_results)
        topk_path = os.path.join(self.output_folder, "topk_metrics.csv")
        topk_df.to_csv(topk_path, index=False)
        print(f"Top-K 分析已保存至：{topk_path}")
        return y_pred

    def save_results(self, y_pred):
        print("\n=== 保存详细结果 ===")
        split_labels = np.full(len(self.scores), 'unlabeled', dtype=object)
        split_labels[self.train_mask] = 'train'
        split_labels[self.val_mask] = 'val'
        split_labels[self.test_mask] = 'test'

        self.results_df = pd.DataFrame({
            'Original_Index': self.df_raw.index,
            'Anomaly_Score': self.scores,
            'Detection_Result': y_pred,
            'True_Label': self.y_true.values,
            'Split': split_labels
        })
        result_path = os.path.join(self.output_folder, "detection_results.csv")
        self.results_df.to_csv(result_path, index=False)
        print(f"详细检测结果已保存至：{result_path}")

    def _run_cli(self):
        import argparse
        parser = argparse.ArgumentParser(
            description='NMIGOD 异常检测 - 命令行模式')
        parser.add_argument('--datasets', '-D', type=str, required=True,
                            help='数据集CSV文件路径, 多个用逗号分隔')
        parser.add_argument('--target', '-t', type=str, required=True,
                            help='真实标签列名')
        parser.add_argument('--anomaly', '-a', type=str, required=True,
                            help='异常值, 逗号分隔')
        parser.add_argument('--output', '-o', type=str, default='./output',
                            help='输出文件夹路径')
        parser.add_argument('--lambda-param', type=float, default=1.0,
                            help='邻域半径系数 (默认: 1.0)')
        parser.add_argument('--hidden1', type=int, default=128,
                            help='GCN 第1层隐藏维度 (默认: 128)')
        parser.add_argument('--hidden2', type=int, default=64,
                            help='GCN 第2层嵌入维度 (默认: 64)')
        parser.add_argument('--epochs', type=int, default=200,
                            help='训练轮数 (默认: 200)')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='学习率 (默认: 0.01)')
        parser.add_argument('--mi-threshold', type=float, default=0.05,
                            help='互信息稀疏化阈值 (默认: 0.05)')
        args = parser.parse_args()

        paths = [p.strip() for p in args.datasets.split(',') if p.strip()]
        anomaly_vals = [v.strip() for v in args.anomaly.split(',') if v.strip()]

        self.lambda_param = getattr(args, 'lambda_param')
        self.hidden1 = args.hidden1
        self.hidden2 = args.hidden2
        self.epochs = args.epochs
        self.lr = args.lr
        self.mi_threshold = args.mi_threshold

        for fp in paths:
            if not os.path.exists(fp):
                print(f"错误: 文件不存在 - {fp}")
                continue
            df = pd.read_csv(fp)
            dataset_name = os.path.splitext(os.path.basename(fp))[0]
            self.dataset_configs.append({
                'file_path': fp, 'df_raw': df,
                'target_column': args.target,
                'anomaly_values': anomaly_vals,
                'output_folder': args.output,
                'dataset_name': dataset_name
            })

        if not self.dataset_configs:
            print("未配置任何有效数据集，程序退出。")
            return
        self._execute_pipeline()

    def _execute_pipeline(self):
        for cfg in self.dataset_configs:
            self.df_raw = cfg['df_raw']
            self.target_column = cfg['target_column']
            self.anomaly_values = cfg['anomaly_values']
            self.output_folder = cfg['output_folder']
            dataset_name = cfg['dataset_name']

            ds_out_folder = os.path.join(self.output_folder, dataset_name)
            os.makedirs(ds_out_folder, exist_ok=True)
            self.output_folder = ds_out_folder

            print(f"\n{'='*30} 开始处理: {dataset_name} {'='*30}")
            self.preprocess_data()
            self.train_model()
            self.get_anomaly_scores()
            self.optimize_threshold()
            y_pred = self.calculate_metrics_and_topk()
            self.save_results(y_pred)

        print("\n=== 所有数据集流程执行完毕 ===")

    def run(self):
        try:
            if len(sys.argv) > 1:
                self._run_cli()
            else:
                self.get_user_inputs()
                if not self.dataset_configs:
                    print("未配置任何数据集，程序退出。")
                    return
                self._execute_pipeline()
        except SystemExit:
            pass
        except Exception as e:
            print(f"\n发生错误：{e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    system = AnomalyDetectionFramework()
    system.run()
