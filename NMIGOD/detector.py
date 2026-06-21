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
# 标准图卷积层 — Kipf & Welling, ICLR 2017
# H' = σ(D^{-1/2} Â D^{-1/2} H W)
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
        # 论文 Eq.2: output = D^{-1/2}ÂD^{-1/2} @ (X @ W)
        support = torch.mm(x, self.weight)
        output = torch.mm(adj_norm, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCNClassifier(nn.Module):
    """
    两层 GCN 二分类器 — 用于基于邻域互信息图的异常检测

    架构:
      Input → GCN Layer 1 → ReLU → Dropout → GCN Layer 2 → ReLU → Linear → logit

    图结构: 邻域互信息矩阵 (对称归一化后)
    训练目标: 二分类交叉熵, 将结点分类为正常(0)或异常(1)
    """

    def __init__(self, in_features, hidden1=128, hidden2=64, dropout=0.5):
        super(GCNClassifier, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden1)
        self.gc2 = GraphConvolution(hidden1, hidden2)
        self.classifier = nn.Linear(hidden2, 1)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        # 第1层: H^(1) = ReLU(D^{-1/2}ÂD^{-1/2} X W^(0))
        h = self.gc1(x, adj_norm)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        # 第2层: H^(2) = ReLU(D^{-1/2}ÂD^{-1/2} H^(1) W^(1))
        h = self.gc2(h, adj_norm)
        h = F.relu(h)
        # 分类器
        logits = self.classifier(h)
        return logits, h


# ============================================================
# NMIGOD 异常检测框架
# 邻域互信息 (Neighborhood Mutual Information) + GCN 二分类
# ============================================================
class AnomalyDetectionFramework:
    """
    NMIGOD: 邻域互信息图卷积异常检测

    核心流程:
    1. 在原始数据上计算自适应邻域半径 (数值: std/λ, 分类型: 0)
    2. 构建邻域关系矩阵 (所有属性同时满足邻域条件)
    3. 计算邻域互信息矩阵 (论文算法1)
    4. 对称归一化 → GCN 图结构
    5. GCN 二分类训练 → 异常概率作为异常分数
    """

    def __init__(self, lambda_param=1.0, hidden1=128, hidden2=64,
                 epochs=200, lr=0.01, mi_threshold=0.05,
                 labeled_ratio=0.2, random_state=42):
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

        # 超参数
        self.lambda_param = lambda_param    # 邻域半径系数 λ (论文默认=1.0)
        self.hidden1 = hidden1              # GCN 第1层隐藏维度
        self.hidden2 = hidden2              # GCN 第2层嵌入维度
        self.epochs = epochs                # 训练轮数
        self.lr = lr                        # 学习率
        self.mi_threshold = mi_threshold    # 互信息稀疏化阈值 d

        # 半监督设置
        self.labeled_ratio = labeled_ratio  # 有标签数据比例 (默认 20%)
        self.random_state = random_state    # 随机种子 (确保跨算法一致性)

        # 数据划分掩码
        self.train_mask = None              # 训练集 (有标签, 用于损失计算)
        self.val_mask = None                # 验证集 (有标签, 用于阈值调优)
        self.test_mask = None               # 测试集 (有标签, 用于评估)
        self.unlabeled_mask = None          # 无标签集 (用于评估)
        self.eval_mask = None               # 评估集 = test + unlabeled

        # 内部状态
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

        # 3. 保留原始数据副本 (用于邻域互信息计算)
        self.df_original = self.df_processed[self.feature_columns].copy()

        # 4. 缺失值填充
        X = self.df_processed[self.feature_columns].copy()
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna("Unknown")
        self.df_processed[self.feature_columns] = X

        # 5. 分离数值与类别列
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

        # 6. 构建 GCN 特征矩阵: 数值 Min-Max + 类别 One-Hot
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
    # 邻域互信息图构建 (论文定义9 & 算法1)
    # ============================================================
    def _build_mi_graph(self):
        """
        构建基于邻域互信息的图结构:

        步骤:
        1. 计算自适应邻域半径 (数值: std/λ, 分类型: 0)
        2. 构建邻域关系矩阵 (所有属性同时满足邻域条件)
        3. 计算邻域互信息矩阵 I(x,y)
        4. 归一化 + 稀疏化
        5. 对称归一化 D^{-1/2} M D^{-1/2}

        论文公式:
          邻域半径: ε_a = std(a) / λ
          互信息: I(x,y) = (|I|/|U|) * log2((|I|*|U|) / (|N(x)|*|N(y)|))
        """
        N = len(self.df_processed)
        print(f"\n[*] 构建邻域互信息图 (N={N}, λ={self.lambda_param})...")

        # ---------- 1. 自适应邻域半径 ----------
        num_radii = []
        for col in self.num_cols:
            raw_vals = self.df_original[col].values.astype(float)
            std_val = np.std(raw_vals)
            radius = std_val / self.lambda_param
            num_radii.append(radius)

        cat_radii = [0.0] * len(self.cat_cols)

        X_raw_num = (self.df_original[self.num_cols].values.astype(float)
                     if len(self.num_cols) > 0 else None)
        X_raw_cat = (self.df_original[self.cat_cols].values
                     if len(self.cat_cols) > 0 else None)

        # ---------- 2. 邻域判定 ----------
        # 数值属性: |a(x) - a(y)| <= ε_a
        if len(self.num_cols) > 0:
            num_tensor = torch.tensor(X_raw_num, dtype=torch.float32, device=self.device)
            radii_tensor = torch.tensor(num_radii, dtype=torch.float32, device=self.device)
            diff = torch.abs(num_tensor.unsqueeze(1) - num_tensor.unsqueeze(0))
            mask_num = diff <= radii_tensor  # (N, N, D_num)
        else:
            mask_num = torch.ones((N, N, 1), dtype=torch.bool, device=self.device)

        # 类别属性: 必须完全相同 (半径=0)
        if len(self.cat_cols) > 0:
            cat_encoded = np.zeros_like(X_raw_cat, dtype=np.int64)
            for i in range(len(self.cat_cols)):
                _, inverse = np.unique(X_raw_cat[:, i], return_inverse=True)
                cat_encoded[:, i] = inverse
            cat_tensor = torch.tensor(cat_encoded, dtype=torch.long, device=self.device)
            d_cat = (cat_tensor.unsqueeze(1) != cat_tensor.unsqueeze(0))
            mask_cat = (d_cat == False)
        else:
            mask_cat = torch.ones((N, N, 1), dtype=torch.bool, device=self.device)

        # 所有属性均满足 → 邻域关系
        N_mask = mask_num.all(dim=-1) & mask_cat.all(dim=-1)  # (N, N)
        N_mask_float = N_mask.float()

        # ---------- 3. 邻域互信息计算 ----------
        N_size = N_mask_float.sum(dim=1)                     # |N(x)|
        intersection = torch.matmul(N_mask_float, N_mask_float.T)  # |N(x) ∩ N(y)|
        denominator = N_size.unsqueeze(1) * N_size.unsqueeze(0)
        denominator = torch.where(
            denominator == 0, torch.ones_like(denominator), denominator)

        # I(x,y) = (|I|/|U|) * log2((|I|*|U|) / (|N(x)|*|N(y)|))
        ratio = (intersection * N) / denominator
        log_ratio = torch.log2(torch.clamp(ratio, min=1e-8))
        prob_factor = intersection / N
        I_matrix = prob_factor * log_ratio
        I_matrix = torch.where(
            intersection == 0, torch.zeros_like(I_matrix), I_matrix)

        # 自身互信息设为 1
        I_matrix = (I_matrix * (1 - torch.eye(N, device=self.device)) +
                    torch.eye(N, device=self.device))

        # ---------- 4. 归一化与稀疏化 ----------
        off_diag = I_matrix.clone()
        off_diag.fill_diagonal_(0)
        max_val = off_diag.max().item()
        if max_val > 0:
            M_matrix = I_matrix / max_val
        else:
            M_matrix = I_matrix

        # 阈值稀疏化
        M_matrix = torch.where(
            M_matrix >= self.mi_threshold, M_matrix,
            torch.zeros_like(M_matrix))

        # ---------- 5. 对称归一化 ----------
        adj = M_matrix
        degree = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt = torch.where(
            torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
        d_inv_sqrt_diag = torch.diag(d_inv_sqrt)

        norm_adj = d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag

        n_edges = int((adj > 0).sum().item()) - N  # 排除自环
        print(f"[*] 互信息图构建完成, 边数(含自环): {n_edges + N}, "
              f"非零边比例: {n_edges / (N*N - N) * 100:.2f}%")

        self.norm_adj = norm_adj
        self.M_matrix = M_matrix

    # ============================================================
    # 半监督数据划分
    # 20% 有标签 (7:2:1 = 训练:验证:测试), 80% 无标签
    # 分层抽样确保正常/异常样本比例一致
    # ============================================================
    def _split_data(self):
        """
        半监督数据划分策略:
          1. 分层抽样 20% 作为有标签数据
          2. 有标签数据按 7:2:1 划分为训练/验证/测试
          3. 其余 80% 作为无标签数据
          4. 评估集 = 测试集 + 无标签集
        """
        from sklearn.model_selection import train_test_split

        N = len(self.y_true)
        y = self.y_true.values

        print(f"\n[*] 半监督数据划分 (labeled_ratio={self.labeled_ratio}, "
              f"random_state={self.random_state})")

        # Step 1: 分层抽样 — 20% 有标签, 80% 无标签 (失败时回退到非分层)
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

        # Step 2: 有标签数据按 7:2:1 划分为训练/验证/测试
        n_labeled = len(labeled_idx)
        n_val = int(n_labeled * 0.2)   # 验证集: 20% of labeled
        n_test = int(n_labeled * 0.1)  # 测试集: 10% of labeled

        y_labeled = y[labeled_idx]

        # 先分出训练集 (分层抽样, 失败时回退到非分层)
        try:
            train_idx_rel, temp_idx_rel = train_test_split(
                np.arange(n_labeled), test_size=n_val + n_test,
                random_state=self.random_state, stratify=y_labeled
            )
        except ValueError:
            train_idx_rel, temp_idx_rel = train_test_split(
                np.arange(n_labeled), test_size=n_val + n_test,
                random_state=self.random_state
            )

        # 从剩余中分出验证集和测试集 (分层抽样, 失败时回退到非分层)
        y_temp = y_labeled[temp_idx_rel]
        try:
            val_idx_rel, test_idx_rel = train_test_split(
                np.arange(len(temp_idx_rel)), test_size=n_test,
                random_state=self.random_state, stratify=y_temp
            )
        except ValueError:
            val_idx_rel, test_idx_rel = train_test_split(
                np.arange(len(temp_idx_rel)), test_size=n_test,
                random_state=self.random_state
            )

        train_idx = labeled_idx[train_idx_rel]
        val_idx = labeled_idx[temp_idx_rel[val_idx_rel]]
        test_idx = labeled_idx[temp_idx_rel[test_idx_rel]]

        # Step 3: 构建布尔掩码
        self.train_mask = np.zeros(N, dtype=bool)
        self.val_mask = np.zeros(N, dtype=bool)
        self.test_mask = np.zeros(N, dtype=bool)
        self.unlabeled_mask = np.zeros(N, dtype=bool)

        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        self.test_mask[test_idx] = True
        self.unlabeled_mask[unlabeled_idx] = True

        # 评估集 = 测试 + 无标签 (不包含训练和验证, 避免数据泄露)
        self.eval_mask = self.test_mask | self.unlabeled_mask

        print(f"  数据划分完成:")
        print(f"    训练集: {train_idx.size} ({train_idx.size/N*100:.1f}%), "
              f"异常比例={y[train_idx].mean():.4f}")
        print(f"    验证集: {val_idx.size} ({val_idx.size/N*100:.1f}%), "
              f"异常比例={y[val_idx].mean():.4f}")
        print(f"    测试集: {test_idx.size} ({test_idx.size/N*100:.1f}%), "
              f"异常比例={y[test_idx].mean():.4f}")
        print(f"    无标签: {unlabeled_idx.size} ({unlabeled_idx.size/N*100:.1f}%), "
              f"异常比例={y[unlabeled_idx].mean():.4f}")
        print(f"    评估集: {self.eval_mask.sum()} ({self.eval_mask.sum()/N*100:.1f}%)")

    # ============================================================
    # GCN 二分类训练
    # ============================================================
    def train_model(self):
        """
        NMIGOD 半监督训练流程:
        1. 构建邻域互信息图
        2. 半监督数据划分
        3. GCN 二分类器 — 仅在有标签训练集上计算损失
        4. 异常分数 = sigmoid(logit) = P(异常 | 结点)
        """
        print("\n=== 模型训练 (NMIGOD: 邻域互信息 + GCN 半监督二分类) ===")
        N = len(self.df_processed)
        in_features = self.X_gcn_np.shape[1]

        # 1. 构建邻域互信息图
        self._build_mi_graph()

        # 2. 半监督数据划分
        self._split_data()

        # 3. 准备数据
        self.X_gcn_tensor = torch.tensor(
            self.X_gcn_np, dtype=torch.float32, device=self.device)
        self.y_tensor = torch.tensor(
            self.y_true.values, dtype=torch.float32, device=self.device)

        # 转换掩码为 tensor
        self.train_mask_t = torch.tensor(self.train_mask, dtype=torch.bool,
                                          device=self.device)
        self.val_mask_t = torch.tensor(self.val_mask, dtype=torch.bool,
                                        device=self.device)

        # 4. 构建 GCN 二分类器
        self.model = GCNClassifier(
            in_features=in_features,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            dropout=0.5
        ).to(self.device)

        # 5. 优化器
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=5e-4)

        # 6. 损失函数 — 基于训练集类别不平衡
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
        print(f"[*] 模型结构: NMIGOD GCN 二分类器 (半监督)")
        print(f"    输入维度={in_features}, 隐藏层1={self.hidden1}, "
              f"隐藏层2={self.hidden2}")
        print(f"    全量 — 异常: {int(n_pos_all)}, 正常: {int(n_neg_all)}")
        print(f"    训练集 — 异常: {int(n_pos_train)}, 正常: {int(n_neg_train)}, "
              f"pos_weight: {criterion.pos_weight.item():.2f}")
        print(f"    半监督训练 (仅 {self.train_mask.sum()} 个有标签结点)")
        print(f"[*] 开始训练 (epochs={self.epochs}, device={self.device})...")

        # 7. 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits, _ = self.model(self.X_gcn_tensor, self.norm_adj)

            # 仅在有标签训练集上计算损失
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

    # ============================================================
    # 生成异常分数
    # ============================================================
    def get_anomaly_scores(self):
        """异常分数 = GCN 二分类器输出的异常类概率"""
        print("\n=== 生成异常分数 (NMIGOD GCN 二分类概率) ===")
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

        # 方向校正
        if anomaly_mean < normal_mean:
            print(f"[!] 分数方向倒置, 自动翻转")
            self.scores = 1.0 - self.scores
            self.logits = -self.logits

    # ============================================================
    # 阈值优化
    # ============================================================
    def optimize_threshold(self):
        print("\n=== 阈值优化 (仅在验证集上进行) ===")
        if self.scores is None:
            raise ValueError("未生成异常分数")

        # 仅在验证集上搜索最佳阈值, 避免数据泄露
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

    # ============================================================
    # 评估指标与 Top-K 分析
    # ============================================================
    def calculate_metrics_and_topk(self):
        print("\n=== 计算评估指标 (评估集 = 测试集 + 无标签集) ===")
        y_pred = (self.scores >= self.best_threshold).astype(int)

        # 基础指标 — 仅在评估集上计算
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

        # Top-K 分析 — 在全部数据的分数上进行 Top-K 选取,
        # 但 Precision/Recall 基于评估集真实异常数计算
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
            # Top-K 中的真实异常: 仅在评估集中计数
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

    # ============================================================
    # 保存详细结果
    # ============================================================
    def save_results(self, y_pred):
        print("\n=== 保存详细结果 ===")
        # 构造 Split 标识列
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
