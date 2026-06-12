import pandas as pd
import numpy as np
import os
import sys
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


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
        self.epsilon = 0.5  # ADFNR 核心参数：模糊邻域半径
        self.numerical_mask = []  # 记录哪些列是数值型

    def get_user_inputs(self):
        print("=== 异常检测系统初始化 ===")
        while True:
            file_path = input("请输入数据集文件路径 (CSV):  ").strip()
            if os.path.exists(file_path):
                break
            else:
                print("文件不存在，请重新输入。")
        self.df_raw = pd.read_csv(file_path)
        print(f"\n数据集加载成功，形状：{self.df_raw.shape}")
        print(f"\n当前数据集列名：\n{list(self.df_raw.columns)}")

        while True:
            target_col = input("\n请输入作为真实标签的异常列名： ").strip()
            if target_col in self.df_raw.columns:
                self.target_column = target_col
                break
            else:
                print("列名不存在，请重新输入。")

        unique_vals = self.df_raw[self.target_column].unique()
        print(f"\n列 '{self.target_column}' 中的唯一值为：{unique_vals}")
        anomaly_input = input("\n请输入代表'异常'的值 (多个值用逗号分隔，例如 1,-1 或 outlier,error):  ").strip()
        self.anomaly_values = [val.strip() for val in anomaly_input.split(',')] if anomaly_input else []

        # 新增：输入 epsilon
        eps_input = input("\n请输入模糊邻域半径 epsilon (范围 0~1，默认 0.5):  ").strip()
        try:
            self.epsilon = float(eps_input) if eps_input else 0.5
        except ValueError:
            self.epsilon = 0.5
        print(f"已设置 epsilon = {self.epsilon}")

        out_folder = input("\n请输入结果保存的文件夹路径 (默认 ./output):  ").strip()
        self.output_folder = out_folder if out_folder else "./output"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"已创建输出文件夹：{self.output_folder}")

    def preprocess_data(self):
        print("\n=== 数据预处理 ===")
        self.df_processed = self.df_raw.copy()

        # 1. 构建真实标签
        def map_anomaly(val):
            if pd.isna(val): return 0
            return 1 if str(val).strip() in self.anomaly_values else 0

        self.y_true = self.df_processed[self.target_column].apply(map_anomaly)

        # 2. 确定特征列
        all_cols = set(self.df_processed.columns)
        drop_cols = {self.target_column}
        self.feature_columns = list(all_cols - drop_cols)
        print(f"用于训练的特征列：{self.feature_columns}")

        # 3. 缺失值处理 & 类型统一/归一化 (为 ADFNR 准备)
        X = self.df_processed[self.feature_columns].copy()
        self.numerical_mask = []
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numerical_mask.append(1)
                # 缺失值填充
                X[col] = X[col].fillna(X[col].mean())
                # 归一化至 [0, 1]
                c_min, c_max = X[col].min(), X[col].max()
                if c_max != c_min:
                    X[col] = (X[col] - c_min) / (c_max - c_min)
                else:
                    X[col] = 0.0
            else:
                self.numerical_mask.append(0)
                # 离散型填众数，并转换为整数编码，便于 ADFNR 矩阵运算
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else "Unknown")
                X[col] = X[col].astype('category').cat.codes

        self.df_processed[self.feature_columns] = X
        self.X_array = X.values.astype(float)  # 统一转为 float 矩阵
        print("缺失值处理、数值归一化与类型转换完成。")

    def train_model(self):
        print("\n=== 模型初始化 ===")
        print("ADFNR 为无监督直接计算算法，跳过传统训练步骤。")
        self.model = "ADFNR"

    def get_anomaly_scores(self):
        print("\n=== 生成异常分数 (ADFNR算法) ===")
        if self.X_array is None:
            raise ValueError("数据未预处理，请先调用 preprocess_data()")
        self.scores = self._compute_adfnr_scores(self.X_array, self.epsilon, self.numerical_mask)
        print(f"异常分数计算完成，分数范围: [{self.scores.min():.4f}, {self.scores.max():.4f}]")

    def _similarity(self, a, x, is_numerical):
        """计算两个属性值的相似度"""
        if is_numerical:
            return 1 - abs(a - x)
        else:
            return 1.0 if a == x else 0.0

    def _compute_adfnr_scores(self, data, epsilon, numerical_mask):
        """核心 ADFNR 算法实现 (内存优化版: float32 + 向量化去重)"""
        n, m = data.shape
        dtype = np.float32  # 使用 float32 节省一半内存

        weight1 = np.zeros((n, m), dtype=dtype)
        weight2 = np.zeros((n, m), dtype=dtype)

        # 1. 批量计算各属性的相似度矩阵 (向量化, 避免 Python 双层循环)
        sim_all = []  # 存储原始相似度矩阵 (float32)
        sim_thresh = []  # 存储阈值化后的矩阵

        for col in range(m):
            vals = data[:, col].astype(dtype)
            if numerical_mask[col]:
                # 数值属性: R = 1 - |a_i - a_j|
                r = 1.0 - np.abs(vals[:, np.newaxis] - vals[np.newaxis, :])
            else:
                # 分类型属性: R = 1 if equal else 0
                r = (vals[:, np.newaxis] == vals[np.newaxis, :]).astype(dtype)
            sim_all.append(r)

            # 阈值化
            r_thresh = r.copy()
            r_thresh[r_thresh < epsilon] = 0.0
            sim_thresh.append(r_thresh)

        # 2. 计算模糊邻域下近似比例与权重
        ratio = np.zeros((n, m), dtype=dtype)

        for col in range(m):
            other_cols = [c for c in range(m) if c != col]
            Set_tem = sim_thresh[col]

            # ---- 内存安全的行去重 ----
            # 关键观察: 分类型属性中, 相同值的对象有完全相同的行模式
            # 因此可以直接按属性值分组, 避免处理整个 n×n 矩阵
            if not numerical_mask[col]:
                # 分类型: 按属性值分组 → 每组对应一个唯一行模式
                attr_vals = data[:, col]
                _, ic = np.unique(attr_vals, return_inverse=True)
                n_vals = ic.max() + 1
                unique_rows = np.zeros((n_vals, n), dtype=dtype)
                for v in range(n_vals):
                    rep_idx = np.where(ic == v)[0][0]
                    unique_rows[v] = Set_tem[rep_idx]
            else:
                # 数值型: 使用 float32 + np.unique(axis=0)
                unique_rows, ic = np.unique(Set_tem.astype(np.float32),
                                            axis=0, return_inverse=True)

            n_unique = len(unique_rows)

            # 计算其他属性相似度矩阵的交集 (T = A - {col})
            Set_tmp = sim_all[other_cols[0]].copy()
            for j in range(1, len(other_cols)):
                np.minimum(Set_tmp, sim_all[other_cols[j]], out=Set_tmp)
            Set_tmp[Set_tmp < epsilon] = 0.0

            # 遍历唯一行计算下近似
            for i in range(n_unique):
                mask = (ic == i)
                row_i = unique_rows[i]

                # 计算下近似: 对所有 k, Set_tmp[k,:] <= row_i
                compare_bool = Set_tmp <= row_i[np.newaxis, :]
                Low_A = int(np.all(compare_bool, axis=1).sum())

                ratio[mask, col] = Low_A / n
                w_val = float(np.sum(row_i)) / n
                weight1[mask, col] = w_val
                weight2[mask, col] = 1.0 - np.power(w_val, 1.0 / 3.0)

        # 释放大矩阵
        del sim_all, sim_thresh

        # 3. 计算异常粒度强度 (GIA) 与最终异常分数 (AS) — 向量化
        GIA = 1.0 - (ratio / m) * weight1
        AS = np.sum(GIA * weight2, axis=1) / m

        return AS.astype(np.float64)

    def optimize_threshold(self):
        print("\n=== 阈值优化 ===")
        if self.scores is None:
            raise ValueError("未生成异常分数")
        best_f1, best_thresh = -1, 0
        thresholds = np.unique(self.scores)
        if len(thresholds) > 100:
            thresholds = np.percentile(self.scores, np.linspace(0, 100, 100))

        for thresh in thresholds:
            y_pred = (self.scores >= thresh).astype(int)
            if np.sum(y_pred) == 0: continue
            try:
                f1 = f1_score(self.y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thresh = f1, thresh
            except Exception:
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
        metrics_path = os.path.join(self.output_folder, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"基础指标已保存至: {metrics_path}")

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

        topk_df = pd.DataFrame(topk_results)
        topk_path = os.path.join(self.output_folder, "topk_metrics.csv")
        topk_df.to_csv(topk_path, index=False)
        print(f"Top-K 分析已保存至: {topk_path}")
        return y_pred

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

    def _run_cli(self):
        """命令行模式: 通过参数直接运行, 无需交互式输入"""
        import argparse
        parser = argparse.ArgumentParser(description='ADFNR 异常检测 - 命令行模式')
        parser.add_argument('--dataset', '-d', type=str, required=True, help='数据集CSV文件路径')
        parser.add_argument('--target', '-t', type=str, required=True, help='真实标签列名')
        parser.add_argument('--anomaly', '-a', type=str, required=True, help='异常值, 逗号分隔 (如 "1,-1")')
        parser.add_argument('--output', '-o', type=str, default='./output', help='输出文件夹路径')
        parser.add_argument('--epsilon', type=float, default=0.5, help='模糊邻域半径 (默认: 0.5)')
        args = parser.parse_args()

        if not os.path.exists(args.dataset):
            print(f"错误: 文件不存在 - {args.dataset}")
            return
        self.df_raw = pd.read_csv(args.dataset)
        self.target_column = args.target
        self.anomaly_values = [v.strip() for v in args.anomaly.split(',') if v.strip()]
        self.output_folder = args.output
        self.epsilon = args.epsilon
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