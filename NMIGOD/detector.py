import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


class SimpleGCN(nn.Module):
    """
    论文中描述的两层图卷积网络 (GCN)
    """

    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleGCN, self).__init__()
        self.gc1 = nn.Linear(in_features, hidden_features)
        self.gc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, adj):
        # Layer 1: 线性变换 -> ReLU -> Dropout -> 图卷积聚合
        h = self.gc1(x)
        h = F.relu(h)
        h = F.dropout(h, 0.5, training=self.training)
        h = torch.matmul(adj, h)

        # Layer 2: 线性变换 -> 图卷积聚合 -> Sigmoid (输出异常概率)
        out = self.gc2(h)
        out = torch.matmul(adj, out)
        return torch.sigmoid(out)


class AnomalyDetectionFramework:
    def __init__(self):
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

        # 用于多数据集管理
        self.dataset_configs = []

        # 设备配置：优先使用 GPU 加速
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] 当前使用设备: {self.device}")

    def get_user_inputs(self):
        print("=== 异常检测系统初始化 (支持多数据集) === ")

        while True:
            file_paths = input("请输入数据集文件路径 (CSV，多个请用逗号分隔):  ").strip()
            paths = [p.strip() for p in file_paths.split(',') if p.strip()]
            if not paths:
                print("未输入有效路径，请重新输入。 ")
                continue
            # 过滤存在的路径
            valid_paths = [p for p in paths if os.path.exists(p)]
            if not valid_paths:
                print("没有找到有效的CSV文件，请重新输入。 ")
                continue
            break

        for fp in valid_paths:
            print(f"\n--- 配置数据集: {os.path.basename(fp)} ---")
            df = pd.read_csv(fp)
            print(f"数据集形状：{df.shape} ")
            print(f"当前列名：{list(df.columns)} ")

            while True:
                target_col = input("请输入作为真实标签的异常列名:  ").strip()
                if target_col in df.columns:
                    break
                print("列名不存在，请重新输入。 ")

            unique_vals = df[target_col].unique()
            print(f"列 '{target_col}' 中的唯一值为：{unique_vals} ")
            anomaly_input = input("请输入代表'异常'的值 (多个用逗号分隔，例如 1,-1 或 outlier,error):  ").strip()
            anomaly_vals = [v.strip() for v in anomaly_input.split(',')] if anomaly_input else []

            out_folder = input("请输入结果保存的文件夹路径 (默认 ./output):  ").strip() or "./output"
            os.makedirs(out_folder, exist_ok=True)

            self.dataset_configs.append({
                'file_path': fp,
                'df_raw': df,
                'target_column': target_col,
                'anomaly_values': anomaly_vals,
                'output_folder': out_folder,
                'dataset_name': os.path.splitext(os.path.basename(fp))[0]
            })

    def preprocess_data(self):
        print("\n=== 数据预处理 === ")
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
        print(f"用于训练的特征列数量：{len(self.feature_columns)} ")

        X = self.df_processed[self.feature_columns].copy()
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna("Unknown")

        self.df_processed[self.feature_columns] = X
        print("缺失值处理完成。 ")

        # ==========================================
        # NMIGOD 专属预处理：分离数值与类别特征，为 HEOM 和 GCN 做准备
        # ==========================================
        self.num_cols = [c for c in self.feature_columns if pd.api.types.is_numeric_dtype(self.df_processed[c])]
        self.cat_cols = [c for c in self.feature_columns if not pd.api.types.is_numeric_dtype(self.df_processed[c])]

        # 1. 数值列 Min-Max 归一化 (论文 4.1 节)
        if len(self.num_cols) > 0:
            X_num = self.df_processed[self.num_cols].values.astype(float)
            self.X_num_min = X_num.min(axis=0)
            self.X_num_max = X_num.max(axis=0)
            # 防止分母为0
            self.X_num_norm = (X_num - self.X_num_min) / (self.X_num_max - self.X_num_min + 1e-8)
        else:
            self.X_num_norm = np.empty((len(self.df_processed), 0))

        # 2. 类别列原始值保存 (用于 HEOM 距离计算) 及 One-Hot 编码 (用于 GCN 节点特征)
        if len(self.cat_cols) > 0:
            self.cat_original = self.df_processed[self.cat_cols].values
            X_cat_oh = pd.get_dummies(self.df_processed[self.cat_cols], dtype=float).values
        else:
            self.cat_original = np.empty((len(self.df_processed), 0))
            X_cat_oh = np.empty((len(self.df_processed), 0))

        # 拼接形成最终 GCN 输入特征矩阵 X
        self.X_gcn_np = np.hstack([self.X_num_norm, X_cat_oh])
        print(f"GCN 特征矩阵维度: {self.X_gcn_np.shape} (数值: {len(self.num_cols)}, 类别One-Hot: {X_cat_oh.shape[1]})")

    def train_model(self):
        print("\n=== 模型训练 (NMIGOD 算法) === ")
        N = len(self.df_processed)
        print(f"[*] 正在 GPU/CPU 上构建邻域互信息矩阵 (规模: {N} x {N})...")

        # ---------------------------------------------------------
        # 步骤 1: 计算 HEOM 距离与邻域矩阵 (算法 1: NMIA)
        # ---------------------------------------------------------
        lambda_param = 1.0  # 论文建议值

        # 1.1 数值属性距离
        if len(self.num_cols) > 0:
            X_num_tensor = torch.tensor(self.X_num_norm, dtype=torch.float32, device=self.device)
            # 使用归一化后的数据计算标准差 (论文定义9)
            sigma = np.std(self.X_num_norm, axis=0)
            sigma = np.where(sigma == 0, 1e-8, sigma)  # 防止常数列导致除零
            sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=self.device)

            # 广播计算 pairwise 差异: (N, 1, D) - (1, N, D) -> (N, N, D)
            diff = torch.abs(X_num_tensor.unsqueeze(1) - X_num_tensor.unsqueeze(0))
            d_num = diff / sigma_tensor
            mask_num = d_num <= lambda_param
        else:
            mask_num = torch.ones((N, N, 1), dtype=torch.bool, device=self.device)

        # 1.2 类别属性距离
        if len(self.cat_cols) > 0:
            cat_encoded = np.zeros_like(self.cat_original, dtype=np.int64)
            for i in range(len(self.cat_cols)):
                _, inverse = np.unique(self.cat_original[:, i], return_inverse=True)
                cat_encoded[:, i] = inverse
            cat_tensor = torch.tensor(cat_encoded, dtype=torch.long, device=self.device)

            # (N, 1, D_cat) != (1, N, D_cat) -> (N, N, D_cat)
            d_cat = (cat_tensor.unsqueeze(1) != cat_tensor.unsqueeze(0))
            mask_cat = (d_cat == False)  # 距离为0表示完全相同
        else:
            mask_cat = torch.ones((N, N, 1), dtype=torch.bool, device=self.device)

        # 1.3 邻域判定 (定义7): 所有属性上的差异均小于等于该属性的半径
        N_mask = mask_num.all(dim=-1) & mask_cat.all(dim=-1)  # (N, N) bool
        N_mask_float = N_mask.float()

        # ---------------------------------------------------------
        # 步骤 2: 计算邻域互信息矩阵 (定义10-14)
        # ---------------------------------------------------------
        N_size = N_mask_float.sum(dim=1)  # |N(x)|, shape: (N,)

        # |N(x) intersect N(y)| 通过矩阵乘法高效计算
        intersection = torch.matmul(N_mask_float, N_mask_float.T)  # (N, N)

        denominator = N_size.unsqueeze(1) * N_size.unsqueeze(0)
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

        # PMI 形式: log2( |N(x) n N(y)| * |U| / (|N(x)| * |N(y)|) )
        ratio = (intersection * N) / denominator
        I_matrix = torch.log2(torch.clamp(ratio, min=1e-8))

        # 当交集为空时，互信息规定为 0
        I_matrix = torch.where(intersection == 0, torch.zeros_like(I_matrix), I_matrix)

        # 归一化 (论文表5到表6的转换规律): M_ij = I_ij / sqrt(I_ii * I_jj)
        I_diag = torch.diag(I_matrix)
        I_diag = torch.clamp(I_diag, min=1e-8)
        denom_norm = torch.sqrt(I_diag.unsqueeze(1) * I_diag.unsqueeze(0))
        M_matrix = I_matrix / denom_norm

        # 规定对象自身的互信息为 1
        M_matrix = M_matrix * (1 - torch.eye(N, device=self.device)) + torch.eye(N, device=self.device)

        # 稀疏化 (定义14): 阈值 d = 0.05 (论文实验设置)
        d_thresh = 0.05
        M_matrix = torch.where(M_matrix >= d_thresh, M_matrix, torch.zeros_like(M_matrix))

        # ---------------------------------------------------------
        # 步骤 3: 构建 GCN 并训练 (算法 2: NMIGOD)
        # ---------------------------------------------------------
        adj = M_matrix

        # 对称归一化: D^(-1/2) * A * D^(-1/2)
        D = torch.diag(adj.sum(dim=1))
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt = torch.where(torch.isinf(D_inv_sqrt), torch.zeros_like(D_inv_sqrt), D_inv_sqrt)
        norm_adj = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)

        # 准备 GCN 输入
        X_gcn_tensor = torch.tensor(self.X_gcn_np, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(self.y_true.values, dtype=torch.float32, device=self.device).unsqueeze(1)

        in_features = X_gcn_tensor.shape[1]
        hidden_features = 64
        out_features = 1

        model = SimpleGCN(in_features, hidden_features, out_features).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 半监督学习掩码：随机选取 20% 的节点作为有标签数据参与损失计算
        labeled_ratio = 0.2
        num_labeled = max(10, int(N * labeled_ratio))  # 至少10个
        indices = torch.randperm(N, device=self.device)
        train_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        train_mask[indices[:num_labeled]] = True

        print(f"[*] 开始半监督训练 (有标签节点数: {num_labeled} / {N})...")
        epochs = 200
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(X_gcn_tensor, norm_adj)

            # 带掩码的二元交叉熵损失
            loss = F.binary_cross_entropy(out[train_mask], y_tensor[train_mask])
            loss.backward()
            optimizer.step()

        print("[*] 模型训练完成。")
        self.model = model
        self.norm_adj = norm_adj
        self.X_gcn_tensor = X_gcn_tensor

    def get_anomaly_scores(self):
        print("\n=== 生成异常分数 === ")
        self.model.eval()
        with torch.no_grad():
            # 模型输出即为属于异常类的概率 (论文定义16)
            scores = self.model(self.X_gcn_tensor, self.norm_adj)
            self.scores = scores.squeeze().cpu().numpy()
        print("异常分数计算完成。")

    def optimize_threshold(self):
        print("\n=== 阈值优化 === ")
        if self.scores is None:
            raise ValueError("未生成异常分数 ")

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
        print(f"最佳阈值：{best_thresh:.4f}, 对应 F1 分数：{best_f1:.4f} ")

    def calculate_metrics_and_topk(self):
        print("\n=== 计算评估指标 === ")
        y_pred = (self.scores >= self.best_threshold).astype(int)

        # 基础指标
        precision = precision_score(self.y_true, y_pred, zero_division=0)
        recall = recall_score(self.y_true, y_pred, zero_division=0)
        f1 = f1_score(self.y_true, y_pred, zero_division=0)

        # 计算 AUC 值
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
        print(f"基础指标已保存至：{metrics_path} ")

        # Top-K 指标计算
        total_count = len(self.scores)

        # 按规则生成 K 值列表
        k_list = []
        # 1% 到 10%，每 1% 计算一次
        for pct in range(1, 11):
            k = max(1, int(total_count * pct / 100))
            k_list.append(k)
        # 10% 到 50%，每 5% 计算一次
        for pct in range(15, 51, 5):
            k = max(1, int(total_count * pct / 100))
            k_list.append(k)
        # 50% 到 100%，每 10% 计算一次
        for pct in range(60, 101, 10):
            k = max(1, int(total_count * pct / 100))
            k_list.append(k)
        # 去重、排序、并确保不超过总数
        k_list = sorted(list(set([min(k, total_count) for k in k_list])))

        # 按分数降序排列索引
        sorted_indices = np.argsort(-self.scores)
        total_true_anomalies = self.y_true.sum()

        topk_results = []

        for k in k_list:
            top_k_indices = sorted_indices[:k]
            y_true_topk = self.y_true.iloc[top_k_indices]
            tp = int(y_true_topk.sum())

            # 计算 Top-K 指标
            prec_k = tp / k if k > 0 else 0.0
            rec_k = tp / total_true_anomalies if total_true_anomalies > 0 else 0.0
            f1_k = 2 * prec_k * rec_k / (prec_k + rec_k) if (prec_k + rec_k) > 0 else 0.0

            topk_results.append({
                'Top_K': k,
                'Percentage(%)': round(k / total_count * 100, 2),
                'Precision': round(prec_k, 4),
                'Recall': round(rec_k, 4),
                'F1-Score': round(f1_k, 4),
                'AUC': round(auc_score, 4),
                'Anomaly_Count_In_TopK': tp
            })

        # 保存 Top-K 详细结果
        topk_df = pd.DataFrame(topk_results)
        topk_path = os.path.join(self.output_folder, "topk_metrics.csv")
        topk_df.to_csv(topk_path, index=False)
        print(f"Top-K 分析已保存至：{topk_path} ")

        return y_pred

    def save_results(self, y_pred):
        print("\n=== 保存详细结果 === ")
        self.results_df = pd.DataFrame({
            'Original_Index': self.df_raw.index,
            'Anomaly_Score': self.scores,
            'Detection_Result': y_pred,
            'True_Label': self.y_true.values
        })
        result_path = os.path.join(self.output_folder, "detection_results.csv")
        self.results_df.to_csv(result_path, index=False)
        print(f"详细检测结果已保存至：{result_path} ")

    def _run_cli(self):
        """命令行模式: 通过参数直接运行, 无需交互式输入 """
        import argparse
        parser = argparse.ArgumentParser(description='通用异常检测框架 (General Framework) - 命令行模式')
        parser.add_argument('--datasets', '-D', type=str, required=True,
                            help='数据集CSV文件路径, 多个用逗号分隔 (如 "data1.csv,data2.csv")')
        parser.add_argument('--target', '-t', type=str, required=True,
                            help='真实标签列名')
        parser.add_argument('--anomaly', '-a', type=str, required=True,
                            help='异常值, 逗号分隔 (如 "1,-1" 或 "outlier,error")')
        parser.add_argument('--output', '-o', type=str, default='./output',
                            help='输出文件夹路径 (默认: ./output)')
        args = parser.parse_args()

        paths = [p.strip() for p in args.datasets.split(',') if p.strip()]
        anomaly_vals = [v.strip() for v in args.anomaly.split(',') if v.strip()]

        for fp in paths:
            if not os.path.exists(fp):
                print(f"错误: 文件不存在 - {fp} ")
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
            print("未配置任何有效数据集，程序退出。 ")
            return
        self._execute_pipeline()

    def _execute_pipeline(self):
        """执行流水线: 遍历所有已配置的数据集 """
        for cfg in self.dataset_configs:
            self.df_raw = cfg['df_raw']
            self.target_column = cfg['target_column']
            self.anomaly_values = cfg['anomaly_values']
            self.output_folder = cfg['output_folder']
            dataset_name = cfg['dataset_name']

            ds_out_folder = os.path.join(self.output_folder, dataset_name)
            os.makedirs(ds_out_folder, exist_ok=True)
            self.output_folder = ds_out_folder

            print(f"\n{'=' * 30} 开始处理: {dataset_name} {'=' * 30} ")
            self.preprocess_data()
            self.train_model()
            self.get_anomaly_scores()
            self.optimize_threshold()
            y_pred = self.calculate_metrics_and_topk()
            self.save_results(y_pred)

        print("\n=== 所有数据集流程执行完毕 === ")

    def run(self):
        try:
            if len(sys.argv) > 1:
                self._run_cli()
            else:
                self.get_user_inputs()
                if not self.dataset_configs:
                    print("未配置任何数据集，程序退出。 ")
                    return
                self._execute_pipeline()
        except SystemExit:
            pass
        except Exception as e:
            print(f"\n发生错误：{e} ")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    system = AnomalyDetectionFramework()
    system.run()