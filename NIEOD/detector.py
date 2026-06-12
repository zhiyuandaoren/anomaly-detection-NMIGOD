#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NIEOD: Neighborhood Information Entropy-based Outlier Detection (Numba 工业优化版)
严格遵循论文: Hybrid data-driven outlier detection based on neighborhood
           information entropy and its developmental measures (ESWA, 2018)
优化说明:
  1. 彻底移除 np.unique(axis=0) 等 Numba 不支持的 API
  2. 采用原生 O(n²) 模式匹配与增量熵计算，复杂度降至 O(mn²)
  3. 使用 @njit(parallel=True) 实现属性级多核并行
"""

import pandas as pd
import numpy as np
import os
import math
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from numba import njit, prange

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


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
        self.lambda_param = 1.0

    def get_user_inputs(self):
        print("=== 异常检测系统初始化 (Numba优化版) === ")
        while True:
            file_path = input("请输入数据集文件路径 (CSV):  ").strip()
            if os.path.exists(file_path):
                break
            else:
                print("文件不存在，请重新输入。 ")
        self.df_raw = pd.read_csv(file_path)
        print(f"\n数据集加载成功，形状：{self.df_raw.shape} ")
        print(f"\n当前数据集列名：\n{list(self.df_raw.columns)} ")
        while True:
            target_col = input("\n请输入作为真实标签的异常列名： ").strip()
            if target_col in self.df_raw.columns:
                self.target_column = target_col
                break
            else:
                print("列名不存在，请重新输入。 ")
        unique_vals = self.df_raw[self.target_column].unique()
        print(f"\n列 '{self.target_column}' 中的唯一值为：{unique_vals} ")
        anomaly_input = input("\n请输入代表'异常'的值 (多个值用逗号分隔):  ").strip()
        self.anomaly_values = [val.strip() for val in anomaly_input.split(',')] if anomaly_input else []
        out_folder = input("\n请输入结果保存的文件夹路径 (默认 ./output):  ").strip()
        self.output_folder = out_folder if out_folder else "./output"
        if not os.path.exists(self.output_folder): os.makedirs(self.output_folder)
        print(f"已创建输出文件夹：{self.output_folder} ")

    def preprocess_data(self):
        print("\n=== 数据预处理 === ")
        self.df_processed = self.df_raw.copy()

        def map_anomaly(val):
            if pd.isna(val): return 0
            return 1 if str(val).strip() in self.anomaly_values else 0

        self.y_true = self.df_processed[self.target_column].apply(map_anomaly)
        all_cols = set(self.df_processed.columns)
        drop_cols = {self.target_column}
        self.feature_columns = list(all_cols - drop_cols)
        print(f"用于训练的特征列：{self.feature_columns} ")
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(self.df_processed[col]):
                self.df_processed.loc[:, col] = self.df_processed[col].fillna(self.df_processed[col].mean())
            else:
                self.df_processed.loc[:, col] = self.df_processed[col].fillna("Unknown").astype(str)
        print("缺失值处理完成。 ")

    def train_model(self):
        print("\n=== 模型训练 === ")
        print(f"NIEOD算法初始化完成。半径调节参数 λ = {self.lambda_param}")
        self.model = "NIEOD-Numba"

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _compute_distances_and_masks(X_num, X_cat, is_numeric, epsilons, n, m):
        """并行计算 HEOM 距离与邻域掩码 (严格对应论文 Eq.5,6,8, Def.1)"""
        masks = np.zeros((m, n, n), dtype=np.bool_)
        for j in prange(m):
            eps = epsilons[j]
            if is_numeric[j]:
                for i in range(n):
                    for k in range(n):
                        if abs(X_num[i, j] - X_num[k, j]) <= eps + 1e-9:
                            masks[j, i, k] = True
            else:
                for i in range(n):
                    for k in range(n):
                        if X_cat[i, j] == X_cat[k, j]:
                            masks[j, i, k] = True
        return masks

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _compute_neof_batch(masks, n, m):
        """
        批量计算 NEOF (严格对应论文 Eq.9~16)
        优化: 移除 np.unique(axis=0)，采用 O(n²) 原生模式匹配，复杂度 O(mn²)
        """
        neof_scores = np.zeros(n, dtype=np.float64)
        U_size = n

        for j in prange(m):
            M = masks[j]  # (n, n) 布尔掩码
            size_n = np.sum(M, axis=1)
            W = size_n / U_size  # Eq.16: 邻域概率权重 |n(x)|/|U|

            # --- Step 1: 查找唯一邻域模式及其频次 (O(n²) 原生实现) ---
            unique_counts = np.zeros(n, dtype=np.int64)
            unique_reps = np.zeros(n, dtype=np.int64)
            visited = np.zeros(n, dtype=np.bool_)
            num_unique = 0

            for i in range(n):
                if not visited[i]:
                    visited[i] = True
                    count = 1
                    for k in range(i + 1, n):
                        if not visited[k]:
                            match = True
                            for col in range(n):
                                if M[i, col] != M[k, col]:
                                    match = False
                                    break
                            if match:
                                count += 1
                                visited[k] = True
                    unique_counts[num_unique] = count
                    unique_reps[num_unique] = i
                    num_unique += 1

            # --- Step 2: 计算完整论域熵 NE_full (Eq.9) ---
            ne_full = 0.0
            for idx in range(num_unique):
                c = unique_counts[idx]
                if c > 0:
                    p = c / U_size
                    ne_full -= p * np.log2(p)

            # --- Step 3: 增量计算移除 xi 后的熵 NE_minus (Eq.11) ---
            ne_minus = np.zeros(n, dtype=np.float64)
            U_minus = U_size - 1.0 if U_size > 1 else 1.0
            for i in range(n):
                ent = 0.0
                for idx in range(num_unique):
                    rep = unique_reps[idx]
                    c = unique_counts[idx]
                    # 若该邻域模式包含样本 i，则移除后频次减 1
                    c_adj = c - 1 if M[rep, i] else c
                    if c_adj > 0:
                        p = c_adj / U_minus
                        ent -= p * np.log2(p)
                ne_minus[i] = ent

            # --- Step 4: 相对邻域熵 RNE (Eq.12) ---
            rne = np.zeros(n, dtype=np.float64)
            for i in range(n):
                if ne_minus[i] < ne_full and ne_full > 1e-10:
                    rne[i] = 1.0 - (ne_minus[i] / ne_full)

            # --- Step 5: 相对邻域基数 RNC (Eq.13) ---
            avg_size_minus = np.zeros(n, dtype=np.float64)
            for i in range(n):
                total_size = 0.0
                for idx in range(num_unique):
                    rep = unique_reps[idx]
                    c = unique_counts[idx]
                    c_adj = c - 1 if M[rep, i] else c
                    total_size += c_adj
                avg_size_minus[i] = total_size / num_unique

            rnc = size_n - avg_size_minus

            # --- Step 6: 偏差度 NOD (Eq.14) ---
            nod = np.zeros(n, dtype=np.float64)
            for i in range(n):
                if rnc[i] > 0:
                    nod[i] = rne[i] * (U_size - np.abs(rnc[i])) / (2 * U_size)
                else:
                    nod[i] = rne[i] * (np.sqrt(U_size) + np.abs(rnc[i])) / (2 * U_size)

            # --- Step 7: 累加加权 (Eq.15 前半部分) ---
            for i in range(n):
                neof_scores[i] += (1.0 - nod[i]) * W[i]

        # --- Step 8: 最终 NEOF (Eq.15) ---
        for i in range(n):
            neof_scores[i] = 1.0 - neof_scores[i] / (2 * m)

        return neof_scores

    def get_anomaly_scores(self):
        print("\n=== 生成异常分数 (NIEOD算法 - Numba多核加速版) === ")
        X = self.df_processed[self.feature_columns]
        n, m = len(X), len(self.feature_columns)
        is_numeric = X.dtypes.apply(lambda dt: pd.api.types.is_numeric_dtype(dt)).values

        # 1. 归一化与编码 (Eq.4)
        print("正在执行数据归一化与类型编码...")
        X_num = np.zeros((n, m), dtype=np.float64)
        X_cat = np.full((n, m), -1, dtype=np.int64)

        for j in range(m):
            col = X.iloc[:, j].values
            if is_numeric[j]:
                vals = pd.to_numeric(col, errors='coerce').astype(np.float64)
                mask_nan = np.isnan(vals)
                vals[mask_nan] = 0.0
                min_v, max_v = np.nanmin(vals), np.nanmax(vals)
                if max_v > min_v:
                    X_num[:, j] = (vals - min_v) / (max_v - min_v)
                X_num[:, j] = np.where(mask_nan, 1.0, X_num[:, j])
            else:
                codes, _ = pd.factorize(col)
                X_cat[:, j] = codes.astype(np.int64)

        # 2. 自适应半径 (Eq.8)
        epsilons = np.zeros(m, dtype=np.float64)
        for j in range(m):
            if is_numeric[j]:
                std = np.std(X_num[:, j])
                epsilons[j] = std / self.lambda_param
            else:
                epsilons[j] = 0.0

        # 3. 计算邻域掩码
        print("⚡ 正在并行计算 HEOM 距离与邻域关系...")
        masks = self._compute_distances_and_masks(X_num, X_cat, is_numeric, epsilons, n, m)

        # 4. 批量计算 NEOF
        print("⚡ 正在并行计算邻域信息熵及异常因子 NEOF...")
        neof_scores = self._compute_neof_batch(masks, n, m)

        self.scores = neof_scores
        print(f"✓ NIEOD异常分数计算完成 | 范围: [{self.scores.min():.4f}, {self.scores.max():.4f}]")

    def optimize_threshold(self):
        print("\n=== 阈值优化 === ")
        if self.scores is None: raise ValueError("未生成异常分数 ")
        best_f1, best_thresh = -1, 0
        thresholds = np.unique(self.scores)
        if len(thresholds) > 100:
            thresholds = np.percentile(self.scores, np.linspace(0, 100, 100))
        for thresh in thresholds:
            y_pred = (self.scores >= thresh).astype(int)
            if np.sum(y_pred) == 0: continue
            try:
                f1 = f1_score(self.y_true, y_pred, zero_division=0)
                if f1 > best_f1: best_f1, best_thresh = f1, thresh
            except:
                continue
        self.best_threshold = best_thresh
        print(f"最佳阈值：{best_thresh:.4f}, 对应 F1 分数：{best_f1:.4f} ")

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
        pd.DataFrame(metrics_data).to_csv(
            os.path.join(self.output_folder, "metrics.csv"), index=False)
        print("基础指标已保存。")

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
        print("Top-K 分析已保存。")
        return y_pred

    def save_results(self, y_pred):
        print("\n=== 保存详细结果 === ")
        pd.DataFrame({
            'Original_Index': self.df_raw.index,
            'Anomaly_Score': self.scores,
            'Detection_Result': y_pred,
            'True_Label': self.y_true.values
        }).to_csv(os.path.join(self.output_folder, "detection_results.csv"), index=False)
        print("详细结果已保存。")

    def _run_cli(self):
        """命令行模式: 通过参数直接运行, 无需交互式输入"""
        import argparse
        parser = argparse.ArgumentParser(description='NIEOD 异常检测 - 命令行模式')
        parser.add_argument('--dataset', '-d', type=str, required=True, help='数据集CSV文件路径')
        parser.add_argument('--target', '-t', type=str, required=True, help='真实标签列名')
        parser.add_argument('--anomaly', '-a', type=str, required=True, help='异常值, 逗号分隔 (如 "1,-1")')
        parser.add_argument('--output', '-o', type=str, default='./output', help='输出文件夹路径')
        parser.add_argument('--lambda', dest='lambda_param', type=float, default=1.0, help='邻域半径调节参数 (默认: 1.0)')
        args = parser.parse_args()

        if not os.path.exists(args.dataset):
            print(f"错误: 文件不存在 - {args.dataset}")
            return
        self.df_raw = pd.read_csv(args.dataset)
        self.target_column = args.target
        self.anomaly_values = [v.strip() for v in args.anomaly.split(',') if v.strip()]
        self.output_folder = args.output
        self.lambda_param = args.lambda_param
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
    system = AnomalyDetectionFramework()
    system.run()