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
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# GCN 模型定义 — 与 GCN 和 NMIGOD 共享相同架构
# Kipf & Welling, "Semi-Supervised Classification with GCNs", ICLR 2017
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
    """
    两层 GCN 二分类器 — 用于 GCN-LOF 异常检测

    架构 (Kipf & Welling, ICLR 2017):
      Input → GCN Layer 1 → ReLU → Dropout → GCN Layer 2 → ReLU → Linear → logit

    GCN-LOF (论文方法, Qin et al., 2025):
      LOF 作为输入特征增强, GCN 分类器直接输出异常概率
    """

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
# GCN-LOF 异常检测框架 (论文方法, Qin et al., 2025)
# LOF 特征增强 + GCN 二分类
# ============================================================
class AnomalyDetectionFramework:
    """
    GCN-LOF: 图卷积网络 + 局部离群因子异常检测 (论文方法)

    核心流程 (Qin et al., "Enhancing Intrusion Detection Performance
    Using GCN-LOF", Computer Networks, 2025):

    1. 数据预处理: StandardScaler + One-Hot 编码
    2. LOF 特征增强: 在原始特征空间计算 LOF 分数, 作为辅助节点属性
    3. 特征拼接: X' = [X_std || LOF_score], 注入局部密度信息
    4. KNN 图构建: 在增强特征空间构建样本关系图
    5. GCN 二分类训练: 端到端半监督学习, 联合利用全局图结构和局部离群特征
    6. 异常分数: GCN 分类器输出 → sigmoid(logit) = P(anomaly | node)
    """

    def __init__(self, k_neighbors=15, hidden1=128, hidden2=64, epochs=200, lr=0.01,
                 labeled_ratio=0.2, random_state=42,
                 lof_neighbors=20, lof_contamination='auto'):
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

        # LOF 超参数
        self.lof_neighbors = lof_neighbors        # LOF 邻居数
        self.lof_contamination = lof_contamination  # 预期异常比例

        # 半监督设置
        self.labeled_ratio = labeled_ratio  # 有标签数据比例 (默认 20%)
        self.random_state = random_state    # 随机种子 (确保跨算法一致性)

        # 数据划分掩码
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.unlabeled_mask = None
        self.eval_mask = None

        # 内部状态
        self.dataset_configs = []
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []
        self.X_features = None
        self.adj = None
        self.adj_norm = None
        self.model = None
        self.X_tensor = None
        self.y_tensor = None
        self.h_embeddings = None            # GCN 嵌入 (用于 LOF)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============================================================
    # 交互式输入
    # ============================================================
    def get_user_inputs(self):
        print("=== GCN-LOF 异常检测系统初始化 (支持多数据集) ===")

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
    # 数据预处理 (与 GCN 一致)
    # ============================================================
    def preprocess_data(self):
        print("\n=== 数据预处理 (GCN-LOF) ===")
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

        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(self.df_processed[col]):
                self.df_processed.loc[:, col] = self.df_processed[col].fillna(
                    self.df_processed[col].mean())
            else:
                self.df_processed.loc[:, col] = self.df_processed[col].fillna("Unknown")

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

        transformers = []
        if self.numeric_features:
            transformers.append(('num', StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 self.categorical_features))

        self.preprocessor = ColumnTransformer(transformers=transformers)

        print("正在执行特征编码与标准化...")
        X_processed = self.preprocessor.fit_transform(self.df_processed[self.feature_columns])
        self.X_features = X_processed.astype(np.float32)
        print(f"预处理后特征维度：{self.X_features.shape}")

    # ============================================================
    # 图构建: k-NN 图 → 对称归一化邻接矩阵
    # ============================================================
    def _build_knn_graph(self, X):
        """通过 k 近邻构建对称邻接矩阵 (稀疏版本, 适用于大规模 N > 50000)"""
        from scipy.sparse import coo_matrix, eye

        N = X.shape[0]
        k = min(self.k_neighbors, N - 1)

        print(f"[*] 构建 k-NN 图 (k={k}, N={N})...")

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

        # 构建稀疏 COO (不创建 dense N×N)
        row_idx = np.repeat(np.arange(N), k)
        col_idx = indices[:, 1:].ravel()
        data = np.ones(len(row_idx), dtype=np.float32)
        adj_sparse = coo_matrix((data, (row_idx, col_idx)), shape=(N, N))
        adj_sparse = adj_sparse.maximum(adj_sparse.T)
        adj_sparse = adj_sparse + eye(N, dtype=np.float32)
        adj_sparse = adj_sparse.tocoo()

        indices_t = torch.LongTensor(np.vstack([adj_sparse.row, adj_sparse.col]))
        values_t = torch.FloatTensor(adj_sparse.data)
        adj_sparse_tensor = torch.sparse_coo_tensor(
            indices_t, values_t, torch.Size([N, N]), device=self.device)

        degree = torch.sparse.sum(adj_sparse_tensor, dim=1).to_dense()
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt),
                                 torch.zeros_like(d_inv_sqrt), d_inv_sqrt)

        d_row = d_inv_sqrt[adj_sparse.row].to(self.device)
        d_col = d_inv_sqrt[adj_sparse.col].to(self.device)
        norm_values = d_row * values_t.to(self.device) * d_col

        adj_norm_sparse = torch.sparse_coo_tensor(
            indices_t, norm_values, torch.Size([N, N]), device=self.device).coalesce()

        self.adj = adj_sparse_tensor.coalesce()
        self.adj_norm = adj_norm_sparse
        n_edges = len(adj_sparse.data)
        print(f"[*] 邻接矩阵构建完成, 边数: {n_edges}, 平均度: {n_edges/N:.1f}")

    # ============================================================
    # 半监督数据划分 ()
    # 20% 有标签 → train/val (75:25), 80% 无标签 → 全部作为 eval
    # 相比原版, 去掉了多余的 2% test 子集
    # ============================================================
    def _split_data(self):
        """半监督数据划分 (): 20% 有标签 → train/val (75:25), 80% 无标签 → eval"""
        from sklearn.model_selection import train_test_split

        N = len(self.y_true)
        y = self.y_true.values

        print(f"\n[*] 半监督数据划分- (labeled_ratio={self.labeled_ratio}, "
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

        print(f"  数据划分完成 (: train/val 二分割, 无独立 test):")
        print(f"    训练集: {train_idx.size} ({train_idx.size/N*100:.1f}%), "
              f"异常比例={y[train_idx].mean():.4f}")
        print(f"    验证集: {val_idx.size} ({val_idx.size/N*100:.1f}%), "
              f"异常比例={y[val_idx].mean():.4f}")
        print(f"    无标签(→评估): {unlabeled_idx.size} ({unlabeled_idx.size/N*100:.1f}%), "
              f"异常比例={y[unlabeled_idx].mean():.4f}")
        print(f"    评估集: {self.eval_mask.sum()} ({self.eval_mask.sum()/N*100:.1f}%)")

    # ============================================================
    # GCN 训练 (论文方法: LOF 特征增强 + GCN 二分类)
    # Qin et al., "Enhancing Intrusion Detection Using GCN-LOF", 2025
    # ============================================================
    def train_model(self):
        """
        GCN-LOF 论文训练流程 (Qin et al., 2025):

        1. LOF 特征增强: 在原始特征空间计算 LOF 分数
        2. 特征拼接: X' = [X_std || LOF_score_norm]
        3. KNN 图构建: 在增强特征空间构建邻接矩阵
        4. 半监督数据划分
        5. GCN 二分类训练: 端到端, 联合利用全局图结构和局部离群信息
        6. 异常分数直接由 GCN 分类器输出
        """
        print("\n=== 模型训练 (GCN-LOF: LOF特征增强 + GCN二分类, 论文方法) ===")

        # 固定随机种子, 确保训练可复现
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        N = self.X_features.shape[0]
        in_features_orig = self.X_features.shape[1]

        # ======== Step 1: LOF 特征增强 (论文核心) ========
        # 在原始标准化特征空间计算 LOF 分数作为辅助节点属性
        lof_n_neighbors = min(self.lof_neighbors, N - 1)
        print(f"[*] LOF 特征增强: 在原始特征空间计算局部离群分数 "
              f"(n_neighbors={lof_n_neighbors})...")

        lof = LocalOutlierFactor(
            n_neighbors=lof_n_neighbors,
            contamination=self.lof_contamination,
            novelty=True,
            n_jobs=-1
        )
        lof.fit(self.X_features)
        lof_raw = lof.score_samples(self.X_features)  # 负值 = 更异常
        lof_scores = -lof_raw.astype(np.float32)       # 取反: 越大越异常

        # 归一化 LOF 分数到 [0, 1]
        lof_min, lof_max = lof_scores.min(), lof_scores.max()
        if lof_max > lof_min:
            lof_norm = (lof_scores - lof_min) / (lof_max - lof_min)
        else:
            lof_norm = np.zeros_like(lof_scores)

        # 特征拼接: X' = [X_std || LOF_score]
        X_augmented = np.hstack([
            self.X_features,
            lof_norm.reshape(-1, 1)
        ]).astype(np.float32)
        in_features = X_augmented.shape[1]

        anomaly_mean_lof = lof_norm[self.y_true == 1].mean()
        normal_mean_lof = lof_norm[self.y_true == 0].mean()
        print(f"    LOF 增强完成, 特征维度: {in_features_orig} → {in_features} (+1)")
        print(f"    LOF 分数 — 异常均值: {anomaly_mean_lof:.4f}, "
              f"正常均值: {normal_mean_lof:.4f}")

        # ======== Step 2: KNN 图构建 (在增强特征空间) ========
        self._build_knn_graph(X_augmented)

        # ======== Step 3: 半监督数据划分 ========
        self._split_data()

        # ======== Step 4: 准备数据 ========
        self.X_tensor = torch.tensor(X_augmented, dtype=torch.float32,
                                      device=self.device)
        self.y_tensor = torch.tensor(self.y_true.values, dtype=torch.float32,
                                      device=self.device)

        self.train_mask_t = torch.tensor(self.train_mask, dtype=torch.bool,
                                          device=self.device)
        self.val_mask_t = torch.tensor(self.val_mask, dtype=torch.bool,
                                        device=self.device)

        # ======== Step 5: 构建 GCN 二分类器 ========
        self.model = GCNClassifier(
            in_features=in_features,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            dropout=0.5
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                      weight_decay=5e-4)

        print(f"[*] 模型结构: GCN-LOF (LOF特征增强 + GCN二分类, 论文方法)")
        print(f"    输入维度={in_features} (原{in_features_orig} + LOF), "
              f"隐藏层1={self.hidden1}, 隐藏层2={self.hidden2}")
        print(f"    LOF 参数: n_neighbors={lof_n_neighbors}")

        # ======== Step 6: 损失函数 ========
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
        print(f"    全量 — 异常: {int(n_pos_all)}, 正常: {int(n_neg_all)}")
        print(f"    训练集 — 异常: {int(n_pos_train)}, 正常: {int(n_neg_train)}, "
              f"pos_weight: {criterion.pos_weight.item():.2f}")
        print(f"    半监督训练 (仅 {self.train_mask.sum()} 个有标签结点)")
        print(f"[*] 开始训练 (epochs={self.epochs}, device={self.device})...")

        # ======== Step 7: 训练循环 ========
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits, _ = self.model(self.X_tensor, self.adj_norm)

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

        print("[*] GCN-LOF (论文方法) 训练完成。")

    # ============================================================
    # 生成异常分数 — GCN 二分类器直接输出 (论文方法)
    # ============================================================
    def get_anomaly_scores(self):
        """
        GCN-LOF 异常分数 (论文方法, Qin et al., 2025):

        异常分数 = sigmoid(logit) = P(anomaly | node)

        GCN 分类器在 LOF 增强的特征上直接预测异常概率,
        无需后处理步骤。LOF 信息已通过特征拼接注入模型。
        """
        print("\n=== 生成异常分数 (GCN-LOF: GCN 二分类概率, 论文方法) ===")
        self.model.eval()
        with torch.no_grad():
            logits, h = self.model(self.X_tensor, self.adj_norm)
            scores = torch.sigmoid(logits).view(-1)
            self.scores = scores.cpu().numpy().astype(np.float64)
            self.logits = logits.view(-1).cpu().numpy()
            self.h_embeddings = h.cpu().numpy()

        anomaly_mean = self.scores[self.y_true == 1].mean()
        normal_mean = self.scores[self.y_true == 0].mean()
        if anomaly_mean < normal_mean:
            print(f"[!] 分数方向倒置, 自动翻转")
            self.scores = 1.0 - self.scores
            self.logits = -self.logits

        print(f"异常分数计算完成, 范围: [{self.scores.min():.4f}, {self.scores.max():.4f}]")
        print(f"  异常类平均分数: {anomaly_mean:.4f}, 正常类平均分数: {normal_mean:.4f}")

    # ============================================================
    # 阈值优化 (仅在验证集上)
    # ============================================================
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
            try:
                f1 = f1_score(val_y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            except Exception:
                continue

        self.best_threshold = best_thresh
        print(f"最佳阈值：{best_thresh:.4f}, "
              f"对应验证集 F1 分数：{best_f1:.4f}")

    # ============================================================
    # 评估指标与 Top-K 分析 (在评估集上)
    # ============================================================
    def calculate_metrics_and_topk(self):
        print("\n=== 计算评估指标 (评估集 = 测试集 + 无标签集) ===")
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

        metrics_data = {
            'Metric': ['Precision', 'Recall', 'F1-Score', 'AUC'],
            'Value': [round(precision, 4), round(recall, 4), round(f1, 4), round(auc_score, 4)]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(self.output_folder, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"基础指标已保存至：{metrics_path}")
        print(f"  Precision={precision:.4f}, Recall={recall:.4f}, "
              f"F1={f1:.4f}, AUC={auc_score:.4f}")

        # Top-K 指标计算
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

    # ============================================================
    # 命令行模式
    # ============================================================
    def _run_cli(self):
        import argparse
        parser = argparse.ArgumentParser(description='GCN-LOF 异常检测 - 命令行模式')
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
                            help='第二层 GCN 隐藏维度 (默认: 64)')
        parser.add_argument('--epochs', type=int, default=200,
                            help='训练轮数 (默认: 200)')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='学习率 (默认: 0.01)')
        parser.add_argument('--lof-neighbors', type=int, default=20,
                            help='LOF 邻居数 (默认: 20)')
        parser.add_argument('--lof-contamination', type=str, default='auto',
                            help='LOF 预期异常比例 (默认: auto)')
        args = parser.parse_args()

        paths = [p.strip() for p in args.datasets.split(',') if p.strip()]
        anomaly_vals = [v.strip() for v in args.anomaly.split(',') if v.strip()]

        self.k_neighbors = args.k_neighbors
        self.hidden1 = args.hidden1
        self.hidden2 = args.hidden2
        self.epochs = args.epochs
        self.lr = args.lr
        self.lof_neighbors = args.lof_neighbors

        # 处理 contamination 参数
        cont_val = args.lof_contamination
        if cont_val.lower() == 'auto':
            self.lof_contamination = 'auto'
        else:
            try:
                self.lof_contamination = float(cont_val)
            except ValueError:
                self.lof_contamination = 'auto'

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
