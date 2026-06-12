import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


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

    def get_user_inputs(self):
        """处理所有用户交互输入"""
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
        if anomaly_input:
            self.anomaly_values = [val.strip() for val in anomaly_input.split(',')]
        else:
            print("未输入异常值，默认将所有非空值视为正常，程序可能无法评估。")
            self.anomaly_values = []

        out_folder = input("\n请输入结果保存的文件夹路径 (默认 ./output):  ").strip()
        self.output_folder = out_folder if out_folder else "./output"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"已创建输出文件夹：{self.output_folder}")

    def preprocess_data(self):
        """数据预处理：缺失值填充，特征与标签分离"""
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

        X = self.df_processed[self.feature_columns]
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna("Unknown")
        self.df_processed[self.feature_columns] = X
        print("缺失值处理完成。")

    def train_model(self):
        """
        基于粗糙集信息熵的异常检测算法为确定性计算，无需传统意义上的模型训练。
        此处保留接口以适配框架流程。
        """
        print("\n=== 模型训练 ===")
        self.model = "IE_RoughSets_OutlierDetector"
        print("已加载基于粗糙集信息熵的异常检测算法 (Düntsch et al. 模型)。")

    def get_anomaly_scores(self):
        """
        严格按照论文 Definition 3.1 ~ 3.7 及 Algorithm 1 实现
        计算每个样本的信息熵异常因子 (EOF)
        """
        print("\n=== 生成异常分数 ===")
        X_df = self.df_processed[self.feature_columns].copy()
        n = len(X_df)
        k = len(X_df.columns)  # |A|
        indices = X_df.index

        # 粗糙集要求基于精确匹配构建不可分辨关系，将特征统一转为字符串
        for col in X_df.columns:
            X_df[col] = X_df[col].astype(str)

        # 1. 计算每个单属性子集的信息熵 E({a}) (Definition 3.1)
        attr_entropies = {}
        for col in X_df.columns:
            counts = X_df[col].value_counts().values
            probs = counts / n
            attr_entropies[col] = -np.sum(np.where(probs > 0, probs * np.log2(probs), 0))

        # 2. 按信息熵升序排列属性 (Definition 3.4)
        sorted_attrs = sorted(attr_entropies.keys(), key=lambda x: attr_entropies[x])
        S = sorted_attrs

        # 3. 构造属性子集降序序列 AS (Definition 3.5)
        AS = []
        current_attrs = list(X_df.columns)
        for j in range(k):
            AS.append(current_attrs.copy())
            if j < k - 1:
                current_attrs = [c for c in current_attrs if c != S[j]]

        # 4. 预计算所有需要处理的子集 (k个单属性 + k个AS子集) 的划分与统计信息
        subsets_to_compute = [(a,) for a in S] + [tuple(asub) for asub in AS]
        subsets_to_compute = list(set(subsets_to_compute))  # 去重
        partition_data = {}

        for sub in subsets_to_compute:
            cols = list(sub)
            # 获取每个样本所属等价类的大小 (对齐原始索引)
            idx_to_size = X_df.groupby(cols).transform('size')
            # 获取所有等价类的大小分布 (用于计算熵和RC)
            class_counts = X_df.groupby(cols).size().values
            probs = class_counts / n
            E_B = -np.sum(np.where(probs > 0, probs * np.log2(probs), 0))

            partition_data[tuple(sub)] = {
                'idx_to_size': idx_to_size.values,  # numpy array 加速索引
                'class_counts': class_counts,
                'E_B': E_B
            }

        # 5. 遍历每个对象计算 EOF (Algorithm 1 核心循环)
        scores = np.zeros(n)

        for i in range(n):
            idx = indices[i]
            numerator_sum = 0.0

            # 定义统一计算块 (用于单属性子集和AS子集)
            def compute_contribution(key):
                nonlocal numerator_sum
                data = partition_data[key]
                E_B = data['E_B']
                c_size = data['idx_to_size'][i]
                all_sizes = data['class_counts']

                # 定位并移除 x 所在等价类
                remove_idx = np.where(all_sizes == c_size)[0][0]
                other_sizes = np.delete(all_sizes, remove_idx)
                m_minus_1 = len(other_sizes)

                # 计算相对熵 RE_B(x) (Definition 3.2)
                RE = 0.0
                if E_B > 0 and m_minus_1 > 0:
                    total_other = n - c_size
                    other_probs = other_sizes / total_other
                    E_x = -np.sum(np.where(other_probs > 0, other_probs * np.log2(other_probs), 0))
                    if E_B > E_x:
                        RE = 1.0 - E_x / E_B

                # 计算相对基数 RC (Definition 3.3)
                if m_minus_1 > 0:
                    avg_other = np.sum(other_sizes) / m_minus_1
                    rc = c_size - avg_other
                else:
                    rc = n  # 仅有一个等价类的情况

                # 计算异常度 OD_B(x) (Definition 3.6)
                abs_rc = abs(rc)
                od = RE * (n - abs_rc) / (2 * n) if rc > 0 else RE * np.sqrt((n + abs_rc) / (2 * n))

                # 计算权重 W_B(x) 并累加
                W = np.sqrt(c_size / n)
                numerator_sum += (1 - od) * W

            # 遍历 k 个单属性子集 {a'_j}
            for a_prime in S:
                compute_contribution((a_prime,))

            # 遍历 k 个属性子集降序序列 A_j
            for asub in AS:
                compute_contribution(tuple(asub))

            # 计算熵异常因子 EOF(x) (Definition 3.7)
            EOF = 1.0 - numerator_sum / (2 * k)
            scores[i] = EOF

        self.scores = scores
        print("异常分数 (EOF) 计算完成。分数越高表示越异常。")

    def optimize_threshold(self):
        """根据 F1 分数自动确定最佳阈值"""
        print("\n=== 阈值优化 ===")
        if self.scores is None:
            raise ValueError("未生成异常分数")
        best_f1 = -1
        best_thresh = 0
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

    def calculate_metrics_and_topk(self):
        """计算评估指标和 Top-K 指标"""
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
        """保存详细检测结果"""
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
        parser = argparse.ArgumentParser(description='IE 信息熵异常检测 - 命令行模式')
        parser.add_argument('--dataset', '-d', type=str, required=True, help='数据集CSV文件路径')
        parser.add_argument('--target', '-t', type=str, required=True, help='真实标签列名')
        parser.add_argument('--anomaly', '-a', type=str, required=True, help='异常值, 逗号分隔 (如 "1,-1")')
        parser.add_argument('--output', '-o', type=str, default='./output', help='输出文件夹路径')
        args = parser.parse_args()

        if not os.path.exists(args.dataset):
            print(f"错误: 文件不存在 - {args.dataset}")
            return
        self.df_raw = pd.read_csv(args.dataset)
        self.target_column = args.target
        self.anomaly_values = [v.strip() for v in args.anomaly.split(',') if v.strip()]
        self.output_folder = args.output
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
        """主流程执行"""
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