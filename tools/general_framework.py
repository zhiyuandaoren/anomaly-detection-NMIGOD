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

        # 用于多数据集管理
        self.dataset_configs = []

    def get_user_inputs(self):
        print("=== 异常检测系统初始化 (支持多数据集) ===")

        while True:
            file_paths = input("请输入数据集文件路径 (CSV，多个请用逗号分隔): ").strip()
            paths = [p.strip() for p in file_paths.split(',') if p.strip()]
            if not paths:
                print("未输入有效路径，请重新输入。")
                continue
            # 过滤存在的路径
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

    def preprocess_data(self):
        print("\n=== 数据预处理 ===")
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
        print(f"用于训练的特征列：{self.feature_columns}")

        X = self.df_processed[self.feature_columns].copy()
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna("Unknown")

        self.df_processed[self.feature_columns] = X
        print("缺失值处理完成。")

    def train_model(self):
        print("\n=== 模型训练 ===")
        X_train = self.df_processed[self.feature_columns].values

        # =========================================================
        # TODO: 在此处实现无监督异常检测算法
        # 示例：from sklearn.ensemble import IsolationForest
        #       self.model = IsolationForest(contamination='auto', random_state=42)
        #       self.model.fit(X_train)
        # =========================================================
        print("正在训练模型 (此处需填入具体算法代码)...")
        self.model = None  # 请在此处初始化您的模型对象

    def get_anomaly_scores(self):
        print("\n=== 生成异常分数 ===")
        X_test = self.df_processed[self.feature_columns].values

        # =========================================================
        # TODO: 在此处使用模型预测异常分数
        # 要求：输出越大约表示越异常
        # 示例：self.scores = -self.model.score_samples(X_test)
        # =========================================================
        print("正在计算异常分数 (此处需填入预测代码)...")
        # 占位符：生成模拟分数用于测试流程
        self.scores = np.random.rand(len(X_test))

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

    def calculate_metrics_and_topk(self):
        print("\n=== 计算评估指标 ===")
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
        print(f"基础指标已保存至：{metrics_path}")

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
        topk_path = os.path.join(self.output_folder, f"topk_metrics.csv")
        topk_df.to_csv(topk_path, index=False)
        print(f"Top-K 分析已保存至：{topk_path}")

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
        parser = argparse.ArgumentParser(
            description='通用异常检测框架 (General Framework) - 命令行模式')
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
        """执行流水线: 遍历所有已配置的数据集"""
        for cfg in self.dataset_configs:
            self.df_raw = cfg['df_raw']
            self.target_column = cfg['target_column']
            self.anomaly_values = cfg['anomaly_values']
            self.output_folder = cfg['output_folder']
            dataset_name = cfg['dataset_name']

            ds_out_folder = os.path.join(self.output_folder, dataset_name)
            os.makedirs(ds_out_folder, exist_ok=True)
            self.output_folder = ds_out_folder

            print(f"\n{'=' * 30} 开始处理: {dataset_name} {'=' * 30}")
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