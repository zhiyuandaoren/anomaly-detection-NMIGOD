import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 【新增】编码与标准化
from sklearn.compose import ColumnTransformer  # 【新增】列转换器


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
        self.k = 10
        self.preprocessor = None  # 【新增】保存预处理器用于后续转换

    def get_user_inputs(self):
        """处理所有用户交互输入（保持不变）"""
        print("=== 异常检测系统初始化 === ")
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
        anomaly_input = input("\n请输入代表'异常'的值 (多个值用逗号分隔，例如 1,-1 或 outlier,error):  ").strip()
        if anomaly_input:
            self.anomaly_values = [val.strip() for val in anomaly_input.split(',')]
        else:
            print("未输入异常值，默认将所有非空值视为正常，程序可能无法评估。 ")
            self.anomaly_values = []
        out_folder = input("\n请输入结果保存的文件夹路径 (默认 ./output):  ").strip()
        self.output_folder = out_folder if out_folder else "./output"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"已创建输出文件夹：{self.output_folder} ")

    def preprocess_data(self):
        """数据预处理：缺失值填充 + 分类编码 + 特征标准化"""
        print("\n=== 数据预处理 === ")
        self.df_processed = self.df_raw.copy()

        # 1. 构建真实标签
        def map_anomaly(val):
            if pd.isna(val):
                return 0
            str_val = str(val).strip()
            return 1 if str_val in self.anomaly_values else 0

        self.y_true = self.df_processed[self.target_column].apply(map_anomaly)

        # 2. 确定训练特征列
        all_cols = set(self.df_processed.columns)
        drop_cols = {self.target_column}
        self.feature_columns = list(all_cols - drop_cols)
        print(f"用于训练的特征列：{self.feature_columns} ")

        # 3. 缺失值填充（使用 .loc 避免 SettingWithCopyWarning）
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(self.df_processed[col]):
                self.df_processed.loc[:, col] = self.df_processed[col].fillna(self.df_processed[col].mean())
            else:
                self.df_processed.loc[:, col] = self.df_processed[col].fillna("Unknown")

        # 4. 【关键修改】分离数值型与分类型特征
        numeric_features = [col for col in self.feature_columns
                            if pd.api.types.is_numeric_dtype(self.df_processed[col])]
        categorical_features = [col for col in self.feature_columns
                                if col not in numeric_features]
        print(f"数值特征 ({len(numeric_features)}): {numeric_features}")
        print(f"分类特征 ({len(categorical_features)}): {categorical_features}")

        # 5. 【关键修改】构建预处理管道：数值型标准化 + 分类型 One-Hot 编码
        transformers = []
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        if categorical_features:
            # handle_unknown='ignore' 防止测试集出现新类别时报错
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))

        self.preprocessor = ColumnTransformer(transformers=transformers)

        # 6. 拟合并转换训练数据
        print("正在执行特征编码与标准化...")
        X_processed = self.preprocessor.fit_transform(self.df_processed[self.feature_columns])
        self.df_processed_encoded = pd.DataFrame(X_processed)  # 保存编码后数据用于训练
        print(f"预处理后特征维度：{X_processed.shape}")
        print("数据预处理完成。")

    def train_model(self):
        """训练 k-NN 模型（严格遵照论文第2、3节）"""
        print("\n=== 模型训练 === ")
        X_train = self.df_processed_encoded.values  # 【修改】使用编码后的数值矩阵

        print(f"正在构建 k-NN 近邻搜索结构 (k={self.k})...")
        # 论文第2节：欧氏距离满足度量标准；第3节：algorithm='auto' 自动选择 KD-Tree/Ball-Tree 加速
        self.model = NearestNeighbors(n_neighbors=self.k, metric='euclidean', algorithm='auto')
        self.model.fit(X_train)
        print("模型训练完成（近邻索引结构构建成功）。")

    def get_anomaly_scores(self):
        """基于论文思想：距离越远越异常"""
        print("\n=== 生成异常分数 === ")
        X_test = self.df_processed_encoded.values

        print("正在计算样本到其 k 个最近邻的距离...")
        distances, _ = self.model.kneighbors(X_test)
        # 使用第 k 个最近邻距离作为异常分数（论文第1节核心直觉的逆向应用）
        self.scores = distances[:, -1]
        print("异常分数计算完成。分数越高表示越异常。")

    def optimize_threshold(self):
        """根据 F1 分数自动确定最佳阈值（保持不变）"""
        print("\n=== 阈值优化 === ")
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
        """保存详细检测结果（保持不变）"""
        print("\n=== 保存详细结果 === ")
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
        parser = argparse.ArgumentParser(description='KNN 异常检测 - 命令行模式')
        parser.add_argument('--dataset', '-d', type=str, required=True, help='数据集CSV文件路径')
        parser.add_argument('--target', '-t', type=str, required=True, help='真实标签列名')
        parser.add_argument('--anomaly', '-a', type=str, required=True, help='异常值, 逗号分隔 (如 "1,-1")')
        parser.add_argument('--output', '-o', type=str, default='./output', help='输出文件夹路径')
        parser.add_argument('--k', type=int, default=10, help='K近邻数 (默认: 10)')
        args = parser.parse_args()

        if not os.path.exists(args.dataset):
            print(f"错误: 文件不存在 - {args.dataset}")
            return
        self.df_raw = pd.read_csv(args.dataset)
        self.target_column = args.target
        self.anomaly_values = [v.strip() for v in args.anomaly.split(',') if v.strip()]
        self.output_folder = args.output
        self.k = args.k
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