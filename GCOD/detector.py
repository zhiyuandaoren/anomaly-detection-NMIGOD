import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetectionFramework:
    def __init__(self, n_jobs=None):
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
        # 使用全部 CPU 核心
        self.n_jobs = n_jobs if n_jobs is not None else cpu_count()

        # FCA-GCOD 内部状态
        self.K_binary = None
        self.E_matrix = None  # (L, U) uint8
        self.extents = []
        self.GCD = None  # (U, U) int16
        self.M_matrix = None  # (L, L) float32
        self.GOD = None  # (L,) float32
        self.U_size = 0
        self.L_size = 0
        self.M_size = 0

    def get_user_inputs(self):
        print("=== 异常检测系统初始化 ===")
        while True:
            file_path = input("请输入数据集文件路径 (CSV):  ").strip()
            if os.path.exists(file_path): break
            print("文件不存在，请重新输入。")
        self.df_raw = pd.read_csv(file_path)
        print(f"\n数据集加载成功，形状：{self.df_raw.shape}")
        print(f"\n当前数据集列名：\n{list(self.df_raw.columns)}")


        while True:
            target_col = input("\n请输入作为真实标签的异常列名： ").strip()
            if target_col in self.df_raw.columns:
                self.target_column = target_col
                break
            print("列名不存在，请重新输入。")

        unique_vals = self.df_raw[self.target_column].unique()
        print(f"\n列 '{self.target_column}' 中的唯一值为：{unique_vals}")

        anomaly_input = input("\n请输入代表'异常'的值 (多个值用逗号分隔):  ").strip()
        self.anomaly_values = [val.strip() for val in anomaly_input.split(',')] if anomaly_input else []
        if not self.anomaly_values: print("未输入异常值，默认将所有非空值视为正常。")

        out_folder = input("\n请输入结果保存的文件夹路径 (默认 ./output):  ").strip()
        self.output_folder = out_folder if out_folder else "./output"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"已创建输出文件夹：{self.output_folder}")

    def preprocess_data(self):
        print("\n=== 数据预处理 ===")
        self.df_processed = self.df_raw.copy()

        def map_anomaly(val):
            if pd.isna(val): return 0
            return 1 if str(val).strip() in self.anomaly_values else 0

        self.y_true = self.df_processed[self.target_column].apply(map_anomaly)

        all_cols = set(self.df_processed.columns)
        drop_cols = {self.target_column}
        self.feature_columns = list(all_cols - drop_cols)
        print(f"用于训练的特征列：{self.feature_columns}")

        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(self.df_processed[col]):
                self.df_processed.loc[:, col] = self.df_processed[col].fillna(self.df_processed[col].mean())
            else:
                self.df_processed.loc[:, col] = self.df_processed[col].fillna("Unknown")
        print("缺失值处理完成。")

    def _build_formal_context(self, X_df):
        """将原始数据离散化并构建二值形式背景 K=(U, M, I)"""
        X_bin = pd.DataFrame(index=X_df.index)
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                if X_df[col].nunique() <= 1:
                    bin_col = pd.DataFrame({f"{col}_bin0": (X_df[col] == X_df[col].iloc[0]).astype(int)},
                                           index=X_df.index)
                    X_bin = pd.concat([X_bin, bin_col], axis=1)
                else:
                    discretizer = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')
                    bin_matrix = discretizer.fit_transform(X_df[[col]])
                    bin_names = [f"{col}_bin{i}" for i in range(5)]
                    X_bin = pd.concat([X_bin, pd.DataFrame(bin_matrix, columns=bin_names, index=X_df.index)], axis=1)
            else:
                dummies = pd.get_dummies(X_df[col], prefix=col, dummy_na=False)
                X_bin = pd.concat([X_bin, dummies.astype(int)], axis=1)
        return X_bin.astype(np.uint8)

    @staticmethod
    def _compute_object_extents_batch(K_vals, indices):
        """批量计算对象粒度概念外延 (多核并行调用)"""
        results = []
        for i in indices:
            row_i = K_vals[i]
            mask = (K_vals >= row_i).all(axis=1)
            results.append(np.where(mask)[0])
        return results

    @staticmethod
    def _compute_attr_extents_batch(K_vals, indices):
        """批量计算属性粒度概念外延 (多核并行调用)"""
        results = []
        for j in indices:
            mask = K_vals[:, j] == 1
            results.append(np.where(mask)[0])
        return results

    def train_model(self):
        print(f"\n=== 模型训练 (FCA-GCOD 极速版 | 并行核心: {self.n_jobs}) ===")
        X_df = self.df_processed[self.feature_columns]
        self.U_size = len(X_df)

        # ---- 步骤1: 离散化并构建二值形式背景 ----
        print("正在离散化并构建二值形式背景 K=(U, M, I)...")
        self.K_binary = self._build_formal_context(X_df)
        self.M_size = len(self.K_binary.columns)
        K_vals = self.K_binary.values
        print(f"形式背景规模: |U|={self.U_size}, |M|={self.M_size}")

        # ---- 步骤2: 并行计算粒度概念外延 ----
        print("正在并行计算粒度概念外延...")

        # 对象外延并行
        obj_chunks = np.array_split(np.arange(self.U_size), self.n_jobs)
        obj_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self._compute_object_extents_batch)(K_vals, chunk)
            for chunk in obj_chunks if len(chunk) > 0
        )
        obj_extents = [ext for batch in obj_results for ext in batch]

        # 属性外延并行
        attr_chunks = np.array_split(np.arange(self.M_size), self.n_jobs)
        attr_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self._compute_attr_extents_batch)(K_vals, chunk)
            for chunk in attr_chunks if len(chunk) > 0
        )
        attr_extents = [ext for batch in attr_results for ext in batch]

        self.extents = obj_extents + attr_extents
        self.L_size = len(self.extents)
        print(f"粒度概念总数 |L| = {self.L_size} (对象: {len(obj_extents)}, 属性: {len(attr_extents)})")

        # ---- 步骤3: 构建 E 矩阵 (L x U) ----
        print("构建 E 矩阵...")
        self.E_matrix = np.zeros((self.L_size, self.U_size), dtype=np.uint8)
        for c_idx, ext in enumerate(self.extents):
            self.E_matrix[c_idx, ext] = 1

        # ---- 步骤4: BLAS 加速计算 GCD ----
        # GCD(x,y) = |{(X,B) in L : x in X xor y in X}|
        #         = row_sum[x] + row_sum[y] - 2 * (E^T @ E)[x,y]
        # 其中 row_sum[x] = 包含对象x的概念数 = E 列求和
        # (E^T @ E)[x,y] = 同时包含对象x和y的概念数
        print(f"正在用 BLAS 矩阵乘法计算 GCD (|U|={self.U_size})...")

        E_float = self.E_matrix.astype(np.float32)  # (L, U)
        E_T = E_float.T  # (U, L) — 不复制数据, 仅改变视图

        # row_sums[i] = 包含对象 i 的概念数量
        row_sums = self.E_matrix.sum(axis=0).astype(np.float32)  # (U,)

        # agreement[i,j] = 同时包含对象i和j的概念数量
        # E_T @ E_float: (U, L) @ (L, U) = (U, U) — 纯 BLAS 矩阵乘法, 极快
        agreement = E_T @ E_float

        # GCD = row_sums[:,None] + row_sums[None,:] - 2 * agreement
        self.GCD = (row_sums[:, None] + row_sums[None, :] - 2.0 * agreement).astype(np.int16)
        del agreement, row_sums
        print("[OK] GCD 矩阵计算完成 (BLAS 加速)。")

        # ---- 步骤5: 矩阵乘法计算 M = (E·GCD·E^T) / (|ext_i|·|ext_j|) ----
        print(f"正在计算概念间距离矩阵 M (|L|={self.L_size})...")
        ext_lens = np.array([len(e) for e in self.extents], dtype=np.float32)

        # EG = E @ GCD : (L, U) @ (U, U) = (L, U) — BLAS 矩阵乘法
        GCD_float = self.GCD.astype(np.float32)
        EG = E_float @ GCD_float  # (L, U)

        # M = EG @ E^T / denom, 分块计算以控制内存
        # 使用 threading 后端 — 对 numpy 操作开销更低
        self.M_matrix = np.zeros((self.L_size, self.L_size), dtype=np.float32)

        # 自适应块大小: 每个核心处理约 L/n_jobs 行, 再分为 4 个子块以平衡负载
        chunk_m = max(50, self.L_size // (self.n_jobs * 4))
        blocks = [(i, min(i + chunk_m, self.L_size)) for i in range(0, self.L_size, chunk_m)]

        with tqdm(total=self.L_size, desc="M 矩阵计算进度", unit="行") as pbar:
            results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(self._compute_m_chunk)(start, end, EG, E_T, ext_lens)
                for start, end in blocks
            )
            for (start, end), block in zip(blocks, results):
                self.M_matrix[start:end, :] = block
                pbar.update(end - start)

        # 对称平均 (处理浮点误差)
        self.M_matrix = (self.M_matrix + self.M_matrix.T) * 0.5
        del E_float, E_T, EG, GCD_float, self.GCD
        print("[OK] M 矩阵计算完成。")

        # ---- 步骤6: 计算 GOD (公式9) ----
        print("正在计算概念异常度 GOD...")
        self.GOD = self.M_matrix.mean(axis=1).astype(np.float32)
        del self.M_matrix
        print("[OK] 模型训练完成 (FCA-GCOD 极速版)。")
        self.model = "FCA_GCOD_Fast"

    def _compute_m_chunk(self, start, end, EG, E_T, ext_lens):
        """
        矩阵乘法计算 M 矩阵行块 [start, end).
        M = (EG @ E^T) / (ext_lens · ext_lens^T)
        """
        denom = ext_lens[start:end, None] * ext_lens[None, :] + 1e-8
        M_block = (EG[start:end, :] @ E_T) / denom
        return M_block.astype(np.float32)

    def get_anomaly_scores(self):
        """论文步骤6：计算对象异常因子 GCOF (公式10)"""
        print("\n=== 生成异常分数 (GCOF) ===")
        sqrt3 = np.float32(np.sqrt(3))

        # 直接复用已构建的 E_matrix (避免重复计算)
        if self.E_matrix is None:
            self.E_matrix = np.zeros((self.L_size, self.U_size), dtype=np.uint8)
            for c_idx, ext in enumerate(self.extents):
                self.E_matrix[c_idx, ext] = 1

        ext_lens_ratio = np.array([len(ext) / self.U_size for ext in self.extents], dtype=np.float32)
        W = 1.0 - sqrt3 * ext_lens_ratio

        numerator = self.E_matrix.T @ (self.GOD * W)
        denominator = self.E_matrix.sum(axis=0).astype(np.float32)  # 沿概念轴求和 (axis=0)
        denominator[denominator == 0] = 1.0

        self.scores = (numerator / denominator).astype(np.float32)
        self.scores = np.nan_to_num(self.scores, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"异常分数 GCOF 计算完成，范围: [{self.scores.min():.4f}, {self.scores.max():.4f}]")

    def optimize_threshold(self):
        print("\n=== 阈值优化 ===")
        if self.scores is None: raise ValueError("未生成异常分数")
        best_f1, best_thresh = -1, 0
        thresholds = np.unique(self.scores)
        if len(thresholds) > 100: thresholds = np.percentile(self.scores, np.linspace(0, 100, 100))
        for thresh in thresholds:
            y_pred = (self.scores >= thresh).astype(int)
            if np.sum(y_pred) == 0: continue
            try:
                f1 = f1_score(self.y_true, y_pred, zero_division=0)
                if f1 > best_f1: best_f1, best_thresh = f1, thresh
            except:
                continue
        self.best_threshold = best_thresh
        print(f"最佳阈值：{best_thresh:.4f}, 对应 F1 分数：{best_f1:.4f}")

    def calculate_metrics_and_topk(self):
        print("\n=== 计算评估指标 ===")
        y_pred = (self.scores >= self.best_threshold).astype(int)
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
        metrics_df.to_csv(os.path.join(self.output_folder, "metrics.csv"), index=False)

        # Top-K 分析 (标准化 K 值计算, 与框架一致)
        total_count = len(self.scores)
        total_true_anomalies = int(self.y_true.sum())

        k_list = []
        for pct in range(1, 11):
            k_list.append(max(1, int(total_count * pct / 100)))
        for pct in range(15, 51, 5):
            k_list.append(max(1, int(total_count * pct / 100)))
        for pct in range(60, 101, 10):
            k_list.append(max(1, int(total_count * pct / 100)))
        k_list = sorted(list(set([min(k, total_count) for k in k_list])))

        topk_results = []
        sorted_indices = np.argsort(-self.scores)
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

        pd.DataFrame(topk_results).to_csv(
            os.path.join(self.output_folder, "topk_metrics.csv"), index=False)
        return y_pred

    def save_results(self, y_pred):
        print("\n=== 保存详细结果 ===")
        self.results_df = pd.DataFrame({
            'Original_Index': self.df_raw.index,
            'Anomaly_Score': self.scores,
            'Detection_Result': y_pred,
            'True_Label': self.y_true.values
        })
        self.results_df.to_csv(os.path.join(self.output_folder, "detection_results.csv"), index=False)
        print(f"结果已保存至：{self.output_folder}")

    def _run_cli(self):
        """命令行模式: 通过参数直接运行, 无需交互式输入"""
        import argparse
        parser = argparse.ArgumentParser(description='GCOD 异常检测 - 命令行模式 (极速版)')
        parser.add_argument('--dataset', '-d', type=str, required=True, help='数据集CSV文件路径')
        parser.add_argument('--target', '-t', type=str, required=True, help='真实标签列名')
        parser.add_argument('--anomaly', '-a', type=str, required=True, help='异常值, 逗号分隔 (如 "1,-1")')
        parser.add_argument('--output', '-o', type=str, default='./output', help='输出文件夹路径')
        parser.add_argument('--n-jobs', type=int, default=None, help='并行核心数 (默认: 全部)')
        args = parser.parse_args()

        if not os.path.exists(args.dataset):
            print(f"错误: 文件不存在 - {args.dataset}")
            return
        self.df_raw = pd.read_csv(args.dataset)
        self.target_column = args.target
        self.anomaly_values = [v.strip() for v in args.anomaly.split(',') if v.strip()]
        self.output_folder = args.output
        if args.n_jobs is not None:
            self.n_jobs = args.n_jobs
        os.makedirs(self.output_folder, exist_ok=True)
        self._execute_pipeline()

    def _execute_pipeline(self):
        self.preprocess_data()
        self.train_model()
        self.get_anomaly_scores()
        self.optimize_threshold()
        y_pred = self.calculate_metrics_and_topk()
        self.save_results(y_pred)
        print("\n=== 流程结束 ===")

    def run(self):
        try:
            if len(sys.argv) > 1:
                self._run_cli()
            else:
                self.get_user_inputs()
                self._execute_pipeline()
        except SystemExit:
            pass
        except Exception as e:
            print(f"\n发生错误：{e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    system = AnomalyDetectionFramework(n_jobs=None)
    system.run()
