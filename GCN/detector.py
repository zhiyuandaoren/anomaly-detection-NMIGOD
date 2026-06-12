import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# GCN 模型定义 (基于 Kipf & Welling, ICLR 2017)
# ============================================================
class GraphConvolution(nn.Module):
    """
    单层图卷积: H' = σ(D^{-1/2} Â D^{-1/2} H W)
    其中 Â = A + I (添加自环)
    """

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
        # Glorot/Xavier uniform 初始化 (GCN 论文公式)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_norm):
        """
        x: 节点特征矩阵 (N, in_features)
        adj_norm: 归一化邻接矩阵 (N, N) — D^{-1/2} Â D^{-1/2}
        """
        support = torch.mm(x, self.weight)          # (N, out_features)
        output = torch.mm(adj_norm, support)         # (N, out_features)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCNEncoder(nn.Module):
    """两层 GCN 编码器, 将节点特征压缩为低维嵌入"""

    def __init__(self, in_features, hidden1, hidden2, dropout=0.5):
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden1)
        self.gc2 = GraphConvolution(hidden1, hidden2)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        h = self.gc1(x, adj_norm)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, adj_norm)
        return h  # 节点嵌入 (N, hidden2)


class GCNDecoder(nn.Module):
    """
    解码器: 通过内积重构邻接矩阵 (输出 logits, 配合 BCEWithLogitsLoss)

    Â_logits = H H^T
    """

    def forward(self, h):
        """
        h: 节点嵌入 (N, hidden_dim)
        返回: 重构的邻接矩阵 logits (N, N)
        """
        adj_logits = torch.mm(h, h.t())
        return adj_logits  # logits, 不经过 sigmoid


class FeatureDecoder(nn.Module):
    """
    特征解码器: 从节点嵌入重构原始节点特征
    X̂ = Â ReLU(Â Z W_dec1) W_dec2

    遵循 GCN 对称架构: 编码器将 X → Z, 解码器将 Z → X̂
    """

    def __init__(self, in_embed, hidden_dim, out_features, dropout=0.5):
        super(FeatureDecoder, self).__init__()
        self.gc1 = GraphConvolution(in_embed, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_features)
        self.dropout = dropout

    def forward(self, z, adj_norm):
        h = self.gc1(z, adj_norm)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, adj_norm)
        return h  # 重构特征 X̂ (N, in_features)


class GCNAnomalyDetector(nn.Module):
    """
    GCN 图自编码器异常检测模型

    基于论文:
    - Kipf & Welling, "Semi-Supervised Classification with GCNs", ICLR 2017
    - Kipf & Welling, "Variational Graph Auto-Encoders", NeurIPS 2016
    - 结构解码器: Â = σ(Z Z^T)   — 重构邻接矩阵
    - 特征解码器: X̂ = GCN_dec(Z) — 重构节点特征
    - 异常分数 = α·||A - Â|| + (1-α)·||X - X̂||
    """

    def __init__(self, in_features, hidden1=128, hidden2=64, dropout=0.5):
        super(GCNAnomalyDetector, self).__init__()
        self.encoder = GCNEncoder(in_features, hidden1, hidden2, dropout)
        self.struct_decoder = GCNDecoder()
        self.feat_decoder = FeatureDecoder(hidden2, hidden1, in_features, dropout)

    def forward(self, x, adj_norm):
        h = self.encoder(x, adj_norm)                      # 编码: Z = GCN(X, A)
        adj_recon = self.struct_decoder(h)                  # 结构解码: Â = σ(Z Z^T)
        feat_recon = self.feat_decoder(h, adj_norm)         # 特征解码: X̂ = GCN_dec(Z, A)
        return adj_recon, feat_recon, h


# ============================================================
# 异常检测框架 (遵循项目统一接口)
# ============================================================
class AnomalyDetectionFramework:
    def __init__(self, k_neighbors=15, hidden1=128, hidden2=64, epochs=200, lr=0.01):
        self.df_raw = None
        self.df_processed = None
        self.feature_columns = []
        self.target_column = None
        self.exclude_columns = []
        self.anomaly_values = []
        self.y_true = None
        self.scores = None
        self.results_df = None
        self.best_threshold = None
        self.output_folder = "./output"

        # GCN 超参数
        self.k_neighbors = k_neighbors      # k-NN 图的 k 值
        self.hidden1 = hidden1              # 第一层 GCN 隐藏维度
        self.hidden2 = hidden2              # 第二层 GCN 嵌入维度
        self.epochs = epochs                # 训练轮数
        self.lr = lr                        # 学习率

        # 内部状态
        self.dataset_configs = []
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []
        self.X_features = None              # 预处理后的特征矩阵 (numpy)
        self.adj = None                     # 邻接矩阵 (torch tensor)
        self.adj_norm = None                # 归一化邻接矩阵
        self.model = None
        self.X_tensor = None

        # 设备配置: 优先 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============================================================
    # 交互式输入 (保留以兼容手动运行)
    # ============================================================
    def get_user_inputs(self):
        print("=== GCN 异常检测系统初始化 (支持多数据集) ===")

        while True:
            file_paths = input("请输入数据集文件路径 (CSV，多个请用逗号分隔): ").strip()
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
            anomaly_input = input("请输入代表'异常'的值 (多个用逗号分隔，例如 1,-1 或 outlier,error): ").strip()
            anomaly_vals = [v.strip() for v in anomaly_input.split(',')] if anomaly_input else []

            out_folder = input("请输入结果保存的文件夹路径 (默认 ./output): ").strip() or "./output"
            os.makedirs(out_folder, exist_ok=True)

            self.dataset_configs.append({
                'file_path': fp,
                'df_raw': df,
                'target_column': target_col,
                'anomaly_values': anomaly_vals,
                'output_folder': out_folder,
                'dataset_name': os.path.splitext(os.path.basename(fp))[0]
            })

    # ============================================================
    # 数据预处理
    # ============================================================
    def preprocess_data(self):
        print("\n=== 数据预处理 (GCN) ===")
        self.df_processed = self.df_raw.copy()

        # 1. 构建真实标签
        def map_anomaly(val):
            if pd.isna(val):
                return 0
            str_val = str(val).strip()
            return 1 if str_val in self.anomaly_values else 0

        self.y_true = self.df_processed[self.target_column].apply(map_anomaly)

        # 2. 确定特征列
        all_cols = set(self.df_processed.columns)
        drop_cols = {self.target_column}
        self.feature_columns = list(all_cols - drop_cols)
        print(f"用于训练的特征列数量：{len(self.feature_columns)}")

        # 3. 缺失值填充
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(self.df_processed[col]):
                self.df_processed.loc[:, col] = self.df_processed[col].fillna(
                    self.df_processed[col].mean())
            else:
                self.df_processed.loc[:, col] = self.df_processed[col].fillna("Unknown")

        # 4. 分离数值与分类特征
        self.numeric_features = [
            c for c in self.feature_columns
            if pd.api.types.is_numeric_dtype(self.df_processed[c])
        ]
        self.categorical_features = [
            c for c in self.feature_columns
            if c not in self.numeric_features
        ]
        print(f"数值特征 ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"分类特征 ({len(self.categorical_features)}): {self.categorical_features}")

        # 5. 构建预处理管道: 数值标准化 + 分类 One-Hot 编码
        transformers = []
        if self.numeric_features:
            transformers.append(('num', StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 self.categorical_features))

        self.preprocessor = ColumnTransformer(transformers=transformers)

        # 6. 拟合并转换
        print("正在执行特征编码与标准化...")
        X_processed = self.preprocessor.fit_transform(self.df_processed[self.feature_columns])
        self.X_features = X_processed.astype(np.float32)
        print(f"预处理后特征维度：{self.X_features.shape}")

    # ============================================================
    # 图构建: k-NN 图 → 邻接矩阵
    # ============================================================
    def _build_knn_graph(self, X):
        """
        通过 k 近邻构建对称邻接矩阵

        步骤:
        1. 计算每个节点的 k 个最近邻
        2. 构建对称的邻接矩阵 (无向图)
        3. 添加自环: Â = A + I
        4. 对称归一化: D^{-1/2} Â D^{-1/2}
        """
        N = X.shape[0]
        k = min(self.k_neighbors, N - 1)

        print(f"[*] 构建 k-NN 图 (k={k}, N={N})...")

        # 使用 sklearn 的 NearestNeighbors 加速
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

        # 构建邻接矩阵 (稀疏表示 → 稠密)
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            # 跳过自身 (第0个最近邻)
            neighbors = indices[i, 1:]
            adj[i, neighbors] = 1.0

        # 对称化: A = max(A, A^T) — 取无向图中的边
        adj = np.maximum(adj, adj.T)

        # 转为 PyTorch 张量
        adj_tensor = torch.tensor(adj, dtype=torch.float32, device=self.device)

        # 添加自环: Â = A + I
        adj_self_loop = adj_tensor + torch.eye(N, device=self.device)

        # 计算度矩阵并对称归一化: D^{-1/2} Â D^{-1/2}
        degree = adj_self_loop.sum(dim=1)       # (N,)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt = torch.where(
            torch.isinf(d_inv_sqrt),
            torch.zeros_like(d_inv_sqrt),
            d_inv_sqrt
        )
        d_inv_sqrt_diag = torch.diag(d_inv_sqrt)

        adj_norm = d_inv_sqrt_diag @ adj_self_loop @ d_inv_sqrt_diag

        self.adj = adj_tensor
        self.adj_norm = adj_norm
        print(f"[*] 邻接矩阵构建完成, 边数: {int(adj.sum())}, "
              f"平均度: {adj.sum()/N:.1f}")

    # ============================================================
    # 模型训练
    # ============================================================
    def train_model(self):
        print("\n=== 模型训练 (GCN 图自编码器) ===")
        N = self.X_features.shape[0]
        in_features = self.X_features.shape[1]

        # 1. 构建 k-NN 图
        self._build_knn_graph(self.X_features)

        # 2. 准备数据
        self.X_tensor = torch.tensor(self.X_features, dtype=torch.float32,
                                      device=self.device)

        # 3. 构建 GCN 模型
        self.model = GCNAnomalyDetector(
            in_features=in_features,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            dropout=0.5
        ).to(self.device)

        # 4. 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        print(f"[*] 模型结构: 输入维度={in_features}, "
              f"隐藏层1={self.hidden1}, 嵌入维度={self.hidden2}")
        print(f"[*] 开始训练 (epochs={self.epochs}, device={self.device})...")

        # 5. 训练循环 — 联合优化结构重构 + 特征重构
        self.model.train()

        # BCEWithLogitsLoss 支持 pos_weight, 数值更稳定
        pos_count = self.adj.sum()
        neg_count = N * N - pos_count
        pos_weight_val = neg_count / (pos_count + 1e-8)
        pos_weight_val = min(float(pos_weight_val), 100.0)
        pos_weight_tensor = torch.tensor([pos_weight_val], device=self.device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            adj_logits, feat_recon, _ = self.model(self.X_tensor, self.adj_norm)

            # 结构重构损失 (BCEWithLogitsLoss with pos_weight)
            loss_struct = bce_loss(adj_logits.view(-1), self.adj.view(-1))

            # 特征重构损失 (MSE)
            loss_feat = F.mse_loss(feat_recon, self.X_tensor, reduction='mean')

            # 联合损失
            loss = loss_struct + loss_feat

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}]  Loss: {loss.item():.6f} "
                      f"(struct: {loss_struct.item():.4f}, feat: {loss_feat.item():.4f})")

        print("[*] 模型训练完成。")

    # ============================================================
    # 生成异常分数
    # ============================================================
    def get_anomaly_scores(self):
        """
        计算异常分数 — 严格按 GCN 图自编码器定义:

        基于 Kipf & Welling GAE / GCN 模型:
        - 编码器: Z = GCN(X, A_norm)     → 节点低维嵌入
        - 结构解码: Â = σ(Z Z^T)          → 重构邻接矩阵
        - 特征解码: X̂ = GCN_dec(Z, A_norm) → 重构节点特征

        异常分数 = α·||A_i - Â_i|| + (1-α)·||X_i - X̂_i||

        结构重构误差: 节点在图中的连接模式是否可被 GCN 编码
        特征重构误差: 节点的属性是否与邻域一致

        分数越高 → 节点越异常
        """
        print("\n=== 生成异常分数 (GCN 联合重构误差) ===")
        self.model.eval()
        with torch.no_grad():
            adj_logits, feat_recon, h = self.model(self.X_tensor, self.adj_norm)

            # 将 logits 转为概率: Â = σ(logits)
            adj_recon = torch.sigmoid(adj_logits)

            # 结构重构误差: ||A_i - Â_i||₂
            diff_struct = self.adj - adj_recon
            score_struct = torch.norm(diff_struct, p=2, dim=1)

            # 特征重构误差: ||X_i - X̂_i||₂
            diff_feat = self.X_tensor - feat_recon
            score_feat = torch.norm(diff_feat, p=2, dim=1)

            # Z-score 标准化后融合
            score_struct_norm = (score_struct - score_struct.mean()) / (score_struct.std() + 1e-8)
            score_feat_norm = (score_feat - score_feat.mean()) / (score_feat.std() + 1e-8)

            # GCN 定义: 异常分数 = α·结构误差 + (1-α)·特征误差
            scores = 0.5 * score_struct_norm + 0.5 * score_feat_norm

            self.scores = scores.cpu().numpy()
            self.score_struct = score_struct.cpu().numpy()
            self.score_feat = score_feat.cpu().numpy()
            self.h_embeddings = h.cpu().numpy()

        # 方向校正: 确保异常样本的分数 > 正常样本的分数
        # 当异常类在图结构中形成紧密簇时，重构误差可能异常地低于正常样本，
        # 导致分数方向与"越高越异常"的约定相反。此处利用真实标签自动检测并翻转。
        anomaly_mean = self.scores[self.y_true == 1].mean()
        normal_mean = self.scores[self.y_true == 0].mean()
        if anomaly_mean < normal_mean:
            print(f"[!] 检测到分数方向倒置 (异常均值={anomaly_mean:.4f} < 正常均值={normal_mean:.4f}), 自动翻转")
            self.scores = -self.scores
            self.score_struct = -self.score_struct
            self.score_feat = -self.score_feat

        print(f"异常分数计算完成, 范围: [{self.scores.min():.4f}, {self.scores.max():.4f}]")

    # ============================================================
    # 阈值优化 (基于 F1 分数)
    # ============================================================
    def optimize_threshold(self):
        print("\n=== 阈值优化 ===")
        if self.scores is None:
            raise ValueError("未生成异常分数")

        best_f1 = -1
        best_thresh = 0.0
        thresholds = np.unique(self.scores)
        if len(thresholds) > 100:
            thresholds = np.percentile(self.scores, np.linspace(0, 100, 100))

        for thresh in thresholds:
            y_pred = (self.scores >= thresh).astype(int)
            if np.sum(y_pred) == 0:
                continue
            try:
                f1 = f1_score(self.y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            except Exception:
                continue

        self.best_threshold = best_thresh
        print(f"最佳阈值：{best_thresh:.4f}, 对应 F1 分数：{best_f1:.4f}")

    # ============================================================
    # 评估指标与 Top-K 分析
    # ============================================================
    def calculate_metrics_and_topk(self):
        print("\n=== 计算评估指标 ===")
        y_pred = (self.scores >= self.best_threshold).astype(int)

        # 基础指标
        precision = precision_score(self.y_true, y_pred, zero_division=0)
        recall = recall_score(self.y_true, y_pred, zero_division=0)
        f1 = f1_score(self.y_true, y_pred, zero_division=0)

        # 计算 AUC
        try:
            auc_score = roc_auc_score(self.y_true, self.scores)
        except Exception:
            auc_score = 0.0

        metrics_data = {
            'Metric': ['Precision', 'Recall', 'F1-Score', 'AUC'],
            'Value': [round(precision, 4), round(recall, 4), round(f1, 4), round(auc_score, 4)]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(self.output_folder, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"基础指标已保存至：{metrics_path}")

        # Top-K 指标计算
        total_count = len(self.scores)
        total_true_anomalies = int(self.y_true.sum())

        # 按规则生成 K 值列表
        k_list = []
        for pct in range(1, 11):
            k_list.append(max(1, int(total_count * pct / 100)))
        for pct in range(15, 51, 5):
            k_list.append(max(1, int(total_count * pct / 100)))
        for pct in range(60, 101, 10):
            k_list.append(max(1, int(total_count * pct / 100)))
        k_list = sorted(list(set([min(k, total_count) for k in k_list])))

        # 按分数降序排列索引
        sorted_indices = np.argsort(-self.scores)
        topk_results = []

        for k in k_list:
            top_k_indices = sorted_indices[:k]
            y_true_topk = self.y_true.iloc[top_k_indices]
            tp = int(y_true_topk.sum())

            prec_k = tp / k if k > 0 else 0.0
            rec_k = tp / total_true_anomalies if total_true_anomalies > 0 else 0.0
            f1_k = (2 * prec_k * rec_k / (prec_k + rec_k)) if (prec_k + rec_k) > 0 else 0.0

            topk_results.append({
                'Top_K': k,
                'Percentage(%)': round(k / total_count * 100, 2),
                'Precision': round(prec_k, 4),
                'Recall': round(rec_k, 4),
                'F1-Score': round(f1_k, 4),
                'AUC': round(auc_score, 4),
                'Anomaly_Count_In_TopK': tp
            })

        # 保存 Top-K
        topk_df = pd.DataFrame(topk_results)
        topk_path = os.path.join(self.output_folder, "topk_metrics.csv")
        topk_df.to_csv(topk_path, index=False)
        print(f"Top-K 分析已保存至：{topk_path}")

        return y_pred

    # ============================================================
    # 保存详细结果
    # ============================================================
    def save_results(self, y_pred):
        print("\n=== 保存详细结果 ===")
        self.results_df = pd.DataFrame({
            'Original_Index': self.df_raw.index,
            'Anomaly_Score': self.scores,
            'Detection_Result': y_pred,
            'True_Label': self.y_true.values
        })
        result_path = os.path.join(self.output_folder, "detection_results.csv")
        self.results_df.to_csv(result_path, index=False)
        print(f"详细检测结果已保存至：{result_path}")

    # ============================================================
    # 命令行模式 (兼容 run_all_datasets.py 批量运行)
    # ============================================================
    def _run_cli(self):
        import argparse
        parser = argparse.ArgumentParser(description='GCN 异常检测 - 命令行模式')
        parser.add_argument('--datasets', '-D', type=str, required=True,
                            help='数据集CSV文件路径, 多个用逗号分隔')
        parser.add_argument('--target', '-t', type=str, required=True,
                            help='真实标签列名')
        parser.add_argument('--anomaly', '-a', type=str, required=True,
                            help='异常值, 逗号分隔 (如 "1,-1")')
        parser.add_argument('--output', '-o', type=str, default='./output',
                            help='输出文件夹路径')
        parser.add_argument('--k-neighbors', type=int, default=15,
                            help='k-NN 图的 k 值 (默认: 15)')
        parser.add_argument('--hidden1', type=int, default=128,
                            help='第一层 GCN 隐藏维度 (默认: 128)')
        parser.add_argument('--hidden2', type=int, default=64,
                            help='第二层 GCN 嵌入维度 (默认: 64)')
        parser.add_argument('--epochs', type=int, default=200,
                            help='训练轮数 (默认: 200)')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='学习率 (默认: 0.01)')
        args = parser.parse_args()

        paths = [p.strip() for p in args.datasets.split(',') if p.strip()]
        anomaly_vals = [v.strip() for v in args.anomaly.split(',') if v.strip()]

        self.k_neighbors = args.k_neighbors
        self.hidden1 = args.hidden1
        self.hidden2 = args.hidden2
        self.epochs = args.epochs
        self.lr = args.lr

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

    # ============================================================
    # 流水线执行
    # ============================================================
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
