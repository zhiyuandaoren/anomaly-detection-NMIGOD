import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

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

        # DASOD 特有参数
        self.K = 5                # 离散化粒度（数值特征等宽区间数）
        self.lambda_ratio = 0.05  # 核心概念选择比例 λ
        self.epsilon = 1e-10      # 避免 log(0)

        # FCA 相关数据结构
        self.formal_context = None     # (U, A, I) 的二进制矩阵 (n_objects x n_attributes)
        self.attr_names = None         # 属性名称列表
        self.object_granular_concepts = None   # 对象粒度概念列表 [(extent_set, intent_set)]
        self.attribute_granular_concepts = None
        self.granular_concepts = None          # 所有粒度概念列表
        self.core_concepts = None              # 核心概念列表
        self.GCD_matrix = None                 # 对象间的 GCD 距离矩阵 (n x n)
        self.M_matrix = None                   # 粒度概念与核心概念之间的 extent 偏差矩阵 (|L_g| x |L_c|)
        self.N_matrix = None                   # intent 偏差矩阵 (|L_g| x |L_c|)
        self.GOD_scores = None                 # 每个粒度概念的 GOD 值
        self.CCOF_scores = None                # 每个对象的 CCOF 值
        self.GPDF_scores = None                # 每个对象的 GPDF 值

        # 多数据集管理
        self.dataset_configs = []

    # ---------- 用户输入 (扩展支持 K 和 λ) ----------
    def get_user_inputs(self):
        print("=== DASOD 异常检测系统初始化 (基于 FCA) ===")
        print("论文: Dual-aspect synergistic outlier detection with structural deviation and attribute rarity\n")

        # 全局参数询问 (对所有数据集通用)
        try:
            k_input = input(f"请输入离散化粒度 K (数值特征等宽区间数, 默认 {self.K}): ").strip()
            if k_input:
                self.K = int(k_input)
            lambda_input = input(f"请输入核心概念选择比例 λ (0~1, 默认 {self.lambda_ratio}): ").strip()
            if lambda_input:
                self.lambda_ratio = float(lambda_input)
        except:
            pass

        while True:
            file_paths = input("\n请输入数据集文件路径 (CSV，多个用逗号分隔): ").strip()
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

    # ---------- 数据预处理 (只做基本的缺失值填充，不改变特征类型) ----------
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

        # 填充缺失值 (数值列用均值，分类列用 "Unknown")
        X = self.df_processed[self.feature_columns].copy()
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna("Unknown")
        self.df_processed[self.feature_columns] = X
        print("缺失值处理完成。")

    # ---------- 核心 DASOD 模型构建 ----------
    def train_model(self):
        print("\n=== 训练 DASOD 模型 (构建形式背景与粒度概念) ===")

        # 1. 将数据离散化为形式背景 (二进制矩阵)
        X = self.df_processed[self.feature_columns]
        n_objects = len(X)
        # 记录每个特征离散化后的区间/类别对应属性名
        attribute_list = []
        bin_mat = []  # 每行是一个对象，每列是一个属性 (0/1)

        for col in self.feature_columns:
            col_data = X[col]
            if pd.api.types.is_numeric_dtype(col_data):
                # 等宽离散化
                min_val = col_data.min()
                max_val = col_data.max()
                if max_val == min_val:
                    # 常数列，只生成一个区间属性
                    intervals = [(min_val, max_val)]
                    attr_names = [f"{col}=[{min_val},{max_val}]"]
                else:
                    width = (max_val - min_val) / self.K
                    intervals = [(min_val + i*width, min_val + (i+1)*width) for i in range(self.K)]
                    attr_names = [f"{col}=({intervals[i][0]:.3f},{intervals[i][1]:.3f}]"
                                  if i>0 else f"{col}=[{intervals[0][0]:.3f},{intervals[0][1]:.3f}]"
                                  for i in range(self.K)]
                # 创建该特征所有区间的 one-hot 编码
                for i, (low, high) in enumerate(intervals):
                    if i == 0:
                        mask = (col_data >= low) & (col_data <= high)
                    else:
                        mask = (col_data > low) & (col_data <= high)
                    bin_mat.append(mask.astype(int))
                    attribute_list.append(attr_names[i])
            else:
                # 分类特征：每个类别一个属性
                categories = col_data.unique()
                for cat in categories:
                    mask = (col_data == cat)
                    bin_mat.append(mask.astype(int))
                    attribute_list.append(f"{col}='{cat}'")

        # 转置得到对象×属性矩阵
        self.formal_context = np.array(bin_mat).T  # shape (n_objects, n_attributes)
        self.attr_names = attribute_list
        n_attrs = self.formal_context.shape[1]
        print(f"形式背景构建完成: {n_objects} 个对象, {n_attrs} 个属性")

        # 移除全0列和全1列（确保正则性）
        col_sum = self.formal_context.sum(axis=0)
        valid_cols = ~((col_sum == 0) | (col_sum == n_objects))
        if not np.all(valid_cols):
            self.formal_context = self.formal_context[:, valid_cols]
            self.attr_names = [self.attr_names[i] for i, v in enumerate(valid_cols) if v]
            print(f"移除了 {np.sum(~valid_cols)} 个全0或全1的属性，剩余属性数: {self.formal_context.shape[1]}")
        # 移除全0行和全1行
        row_sum = self.formal_context.sum(axis=1)
        valid_rows = ~((row_sum == 0) | (row_sum == self.formal_context.shape[1]))
        if not np.all(valid_rows):
            self.formal_context = self.formal_context[valid_rows, :]
            self.y_true = self.y_true[valid_rows]    # 同步标签
            print(f"移除了 {np.sum(~valid_rows)} 个全0或全1的对象，剩余对象数: {self.formal_context.shape[0]}")

        n_objects, n_attrs = self.formal_context.shape
        if n_objects < 2:
            raise ValueError("经过正则化后对象数量不足2，无法进行异常检测")

        # 2. 计算对象粒度概念 L_og 和属性粒度概念 L_ag
        # 定义上算子 (^): X↑ = {a | ∀x∈X, I(x,a)=1}
        # 定义下算子 (↓): B↓ = {x | ∀a∈B, I(x,a)=1}
        def extent_intent_from_objects(X_set):
            # X_set: 对象索引集合 (set)
            # 返回 (extent, intent) 其中 intent 是属性索引集合
            if not X_set:
                return set(), set()
            # 取所有对象共有的属性
            common_attrs = set(range(n_attrs))
            for x in X_set:
                row = set(np.where(self.formal_context[x] == 1)[0])
                common_attrs &= row
            # 再计算闭包: B↓↑
            extent = set()
            for a in common_attrs:
                extent |= set(np.where(self.formal_context[:, a] == 1)[0])
            return extent, common_attrs

        def intent_extent_from_attributes(B_set):
            # B_set: 属性索引集合
            if not B_set:
                return set(), set()
            common_objs = set(range(n_objects))
            for a in B_set:
                col = set(np.where(self.formal_context[:, a] == 1)[0])
                common_objs &= col
            # 闭包: X↑↓
            intent = set()
            for x in common_objs:
                intent |= set(np.where(self.formal_context[x] == 1)[0])
            return common_objs, intent

        # 计算对象粒度概念 (x↑↓, x↑)
        self.object_granular_concepts = []
        for x in range(n_objects):
            # 单对象 extent 的闭包
            obj_set = {x}
            extent, intent = extent_intent_from_objects(obj_set)
            # 注意闭包后 extent 可能变大，但定义中粒度概念就是 (x↑↓, x↑)
            # 计算 x↑ (该对象拥有的所有属性)
            attrs_of_x = set(np.where(self.formal_context[x] == 1)[0])
            # x↑↓ 是拥有 attrs_of_x 中所有属性的对象集合
            extent_closed = set(range(n_objects))
            for a in attrs_of_x:
                extent_closed &= set(np.where(self.formal_context[:, a] == 1)[0])
            self.object_granular_concepts.append((frozenset(extent_closed), frozenset(attrs_of_x)))

        # 属性粒度概念 (a↓, a↓↑)
        self.attribute_granular_concepts = []
        for a in range(n_attrs):
            attrs_set = {a}
            extent, intent = intent_extent_from_attributes(attrs_set)
            # 或者直接计算 a↓
            extent_a = set(np.where(self.formal_context[:, a] == 1)[0])
            # a↓↑ 是 extent_a 中所有对象共有的属性
            intent_closed = set(range(n_attrs))
            for x in extent_a:
                intent_closed &= set(np.where(self.formal_context[x] == 1)[0])
            self.attribute_granular_concepts.append((frozenset(extent_a), frozenset(intent_closed)))

        # 合并所有粒度概念，去重 (使用 frozenset 对)
        all_concepts = []
        seen = set()
        for ext, intt in self.object_granular_concepts + self.attribute_granular_concepts:
            key = (ext, intt)
            if key not in seen:
                seen.add(key)
                all_concepts.append((ext, intt))
        self.granular_concepts = all_concepts
        n_gc = len(self.granular_concepts)
        print(f"生成粒度概念数量: {n_gc} (对象粒 {len(self.object_granular_concepts)} + 属性粒 {len(self.attribute_granular_concepts)}，去重后)")

        # 3. 选择核心概念 (按 extent 支持度降序)
        supports = [len(ext) / n_objects for ext, _ in self.granular_concepts]
        sorted_idx = np.argsort(supports)[::-1]  # 降序
        n_core = max(1, int(self.lambda_ratio * n_gc))
        core_indices = sorted_idx[:n_core]
        self.core_concepts = [self.granular_concepts[i] for i in core_indices]
        print(f"选择核心概念数量: {len(self.core_concepts)} (λ={self.lambda_ratio})")

        # 4. 计算对象间的 GCD 距离矩阵 (n_objects × n_objects)
        print("计算对象间 GCD 距离矩阵...")
        # 构建对象-粒度概念隶属矩阵 (n_objects × n_gc)
        membership = np.zeros((n_objects, n_gc), dtype=bool)
        for i, (ext, _) in enumerate(self.granular_concepts):
            for x in ext:
                membership[x, i] = True
        # GCD(x,y) = 异或求和 (不同属的概念数量)
        # 利用矩阵乘法: GCD = membership XOR membership => 快速计算
        # 技巧: (a XOR b) = a + b - 2*a*b, 对于布尔值
        membership_int = membership.astype(np.int8)
        # 计算点积: dot = membership @ membership.T
        dot = membership_int @ membership_int.T
        sum_row = membership_int.sum(axis=1).reshape(-1, 1)
        gcd_matrix = sum_row + sum_row.T - 2 * dot
        self.GCD_matrix = gcd_matrix  # shape (n_objects, n_objects)

        # 5. 计算每个粒度概念的代表意图向量 (用于 N 度量)
        # 需要原始离散化后的数值特征 (每个对象在每个特征上的离散区间索引)
        # 重新获取离散化后的特征矩阵 (n_objects × d)
        # 为了效率，我们在离散化时保存每个对象在每个特征上的区间编号
        # 修改前面离散化部分，记录区间映射 (这里简单重建)
        print("计算意图向量...")
        # 重新得到特征离散索引表 (n_objects × d)
        feature_discrete = np.zeros((n_objects, len(self.feature_columns)), dtype=int)
        for j, col in enumerate(self.feature_columns):
            col_data = X[col]
            if pd.api.types.is_numeric_dtype(col_data):
                min_val = col_data.min()
                max_val = col_data.max()
                if max_val == min_val:
                    # 常数列，所有对象区间编号 0
                    feature_discrete[:, j] = 0
                else:
                    width = (max_val - min_val) / self.K
                    # 计算每个值所在的区间编号
                    bins = np.linspace(min_val, max_val, self.K+1)
                    bins[0] = -np.inf  # 包含左边界
                    bins[-1] = np.inf
                    indices = np.digitize(col_data, bins[1:-1])  # 返回 1..K
                    feature_discrete[:, j] = indices - 1  # 转为 0..K-1
            else:
                # 分类特征：每个类别映射到唯一编号 (0..n_cat-1)
                unique = list(col_data.unique())
                mapping = {v: i for i, v in enumerate(unique)}
                feature_discrete[:, j] = [mapping[v] for v in col_data]

        # 计算每个粒度概念的意图向量 (长度为 d，若概念内所有对象在某特征上取值相同则取该值，否则 0)
        intent_vectors = []
        for ext, _ in self.granular_concepts:
            ext_list = list(ext)
            if not ext_list:
                vec = np.zeros(len(self.feature_columns), dtype=int)
            else:
                vec = np.zeros(len(self.feature_columns), dtype=int)
                for j in range(len(self.feature_columns)):
                    values = feature_discrete[ext_list, j]
                    if np.all(values == values[0]):
                        vec[j] = values[0] + 1  # 使用 1..K 表示，0 表示不一致
                    else:
                        vec[j] = 0
            intent_vectors.append(vec)

        # 6. 计算 M (extent偏差) 和 N (intent偏差) 矩阵: (n_gc × n_core)
        n_core = len(self.core_concepts)
        M_matrix = np.zeros((n_gc, n_core))
        N_matrix = np.zeros((n_gc, n_core))

        print("计算结构偏差矩阵 M 和 N...")
        # 预先获取每个概念的 extent 大小和 intent 向量
        extents = [ext for ext, _ in self.granular_concepts]
        core_extents = [ext for ext, _ in self.core_concepts]

        # 计算 M: 对于每个粒度概念 C 和每个核心概念 Cc
        # M(C, Cc) = (∑_{x∈X} ∑_{y∈Y} GCD(x,y)) / (|X||Y|)
        for i in range(n_gc):
            ext_i = extents[i]
            size_i = len(ext_i)
            for j, ext_j in enumerate(core_extents):
                size_j = len(ext_j)
                if size_i == 0 or size_j == 0:
                    M_matrix[i, j] = 0.0
                else:
                    # 计算 sum_{x∈ext_i} sum_{y∈ext_j} GCD(x,y)
                    # 使用索引列表
                    idx_i = list(ext_i)
                    idx_j = list(ext_j)
                    # 从 GCD_matrix 中提取子矩阵并求和
                    sub_gcd = self.GCD_matrix[np.ix_(idx_i, idx_j)]
                    total = np.sum(sub_gcd)
                    M_matrix[i, j] = total / (size_i * size_j)

        # 计算 N: intent-based distinction degree
        for i in range(n_gc):
            vec_i = intent_vectors[i]
            for j in range(n_core):
                vec_j = intent_vectors[core_indices[j]]
                # 找到共同有定义的特征 (两向量均非0)
                common = (vec_i != 0) & (vec_j != 0)
                if np.any(common):
                    diff = np.abs(vec_i[common] - vec_j[common])
                    max_diff = np.max(diff)
                    if max_diff > 0:
                        N_matrix[i, j] = max_diff
                    else:
                        N_matrix[i, j] = self.K / 2.0   # 平均距离 (论文公式9)
                else:
                    N_matrix[i, j] = self.K              # 没有共同特征，最大冲突
                # 注: 论文中 K 是离散化粒度，此处用 self.K

        # 7. 计算每个粒度概念的 GOD = (1/|L_c|) * Σ M * N
        self.GOD_scores = np.mean(M_matrix * N_matrix, axis=1)

        # 8. 计算每个对象的 CCOF
        # 对于每个对象 x，找到包含 x 的所有粒度概念索引
        obj_to_concepts = [[] for _ in range(n_objects)]
        for i, (ext, _) in enumerate(self.granular_concepts):
            for x in ext:
                obj_to_concepts[x].append(i)
        self.CCOF_scores = np.zeros(n_objects)
        for x in range(n_objects):
            concepts = obj_to_concepts[x]
            if not concepts:
                self.CCOF_scores[x] = 0.0
                continue
            total = 0.0
            sum_weight = 0.0
            for idx in concepts:
                ext_size = len(extents[idx])
                w = 1 - np.sqrt(ext_size / n_objects)
                total += self.GOD_scores[idx] * w
                sum_weight += w
            if sum_weight > 0:
                self.CCOF_scores[x] = total / sum_weight
            else:
                self.CCOF_scores[x] = 0.0

        # 9. 计算每个对象的 GPDF (全局属性稀有度)
        # 计算每个属性的全局概率 P(a)
        attr_prob = self.formal_context.mean(axis=0)  # 列均值
        # 避免 log(0)
        attr_prob = np.clip(attr_prob, self.epsilon, 1.0)
        self.GPDF_scores = np.zeros(n_objects)
        for x in range(n_objects):
            # 对象 x 拥有的属性索引
            attrs = np.where(self.formal_context[x] == 1)[0]
            if len(attrs) == 0:
                self.GPDF_scores[x] = 0.0
            else:
                info = -np.log(attr_prob[attrs])
                self.GPDF_scores[x] = np.sum(info)

        print("DASOD 模型训练完成 (核心数据结构已构建)")

    # ---------- 计算异常分数 (融合 CCOF 和 GPDF) ----------
    def get_anomaly_scores(self):
        print("\n=== 生成异常分数 (DASOF) ===")
        # 融合因子: DASOF = CCOF * GPDF
        self.scores = self.CCOF_scores * self.GPDF_scores
        # 归一化到 [0,1] 便于比较
        min_s = np.min(self.scores)
        max_s = np.max(self.scores)
        if max_s > min_s:
            self.scores = (self.scores - min_s) / (max_s - min_s)
        else:
            self.scores = np.zeros_like(self.scores)
        print(f"异常分数范围: [{self.scores.min():.4f}, {self.scores.max():.4f}]")

    # ---------- 以下方法与原框架完全一致 ----------
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
            f1 = f1_score(self.y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        self.best_threshold = best_thresh
        print(f"最佳阈值：{best_thresh:.4f}, 对应 F1 分数：{best_f1:.4f}")

    def calculate_metrics_and_topk(self):
        print("\n=== 计算评估指标 ===")
        y_pred = (self.scores >= self.best_threshold).astype(int)
        precision = precision_score(self.y_true, y_pred, zero_division=0)
        recall = recall_score(self.y_true, y_pred, zero_division=0)
        f1 = f1_score(self.y_true, y_pred, zero_division=0)
        try:
            auc_score = roc_auc_score(self.y_true, self.scores)
        except:
            auc_score = 0.0
        metrics_data = {
            'Metric': ['Precision', 'Recall', 'F1-Score', 'AUC'],
            'Value': [round(precision,4), round(recall,4), round(f1,4), round(auc_score,4)]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(self.output_folder, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"基础指标已保存至：{metrics_path}")

        total_count = len(self.scores)
        k_list = []
        for pct in range(1, 11):
            k_list.append(max(1, int(total_count * pct / 100)))
        for pct in range(15, 51, 5):
            k_list.append(max(1, int(total_count * pct / 100)))
        for pct in range(60, 101, 10):
            k_list.append(max(1, int(total_count * pct / 100)))
        k_list = sorted(list(set([min(k, total_count) for k in k_list])))

        sorted_indices = np.argsort(-self.scores)
        total_true_anomalies = self.y_true.sum()
        topk_results = []
        for k in k_list:
            top_k_indices = sorted_indices[:k]
            y_true_topk = self.y_true.iloc[top_k_indices]
            tp = int(y_true_topk.sum())
            prec_k = tp / k if k>0 else 0.0
            rec_k = tp / total_true_anomalies if total_true_anomalies>0 else 0.0
            f1_k = 2 * prec_k * rec_k / (prec_k + rec_k) if (prec_k+rec_k)>0 else 0.0
            topk_results.append({
                'Top_K': k,
                'Percentage(%)': round(k / total_count * 100, 2),
                'Precision': round(prec_k,4),
                'Recall': round(rec_k,4),
                'F1-Score': round(f1_k,4),
                'AUC': round(auc_score,4),
                'Anomaly_Count_In_TopK': tp
            })
        topk_df = pd.DataFrame(topk_results)
        topk_path = os.path.join(self.output_folder, "topk_metrics.csv")
        topk_df.to_csv(topk_path, index=False)
        print(f"Top-K 分析已保存至：{topk_path}")
        return y_pred

    def save_results(self, y_pred):
        print("\n=== 保存详细结果 ===")
        self.results_df = pd.DataFrame({
            'Original_Index': self.df_raw.index[:len(self.scores)],
            'Anomaly_Score': self.scores,
            'Detection_Result': y_pred,
            'True_Label': self.y_true.values
        })
        result_path = os.path.join(self.output_folder, "detection_results.csv")
        self.results_df.to_csv(result_path, index=False)
        print(f"详细检测结果已保存至：{result_path}")

    # ---------- 主流程 (多数据集支持) ----------
    def _run_cli(self):
        """命令行模式: 通过参数直接运行, 无需交互式输入"""
        import argparse
        parser = argparse.ArgumentParser(description='DASOD 异常检测 - 命令行模式')
        parser.add_argument('--datasets', '-D', type=str, required=True,
                            help='数据集CSV文件路径, 多个用逗号分隔')
        parser.add_argument('--target', '-t', type=str, required=True, help='真实标签列名')
        parser.add_argument('--anomaly', '-a', type=str, required=True, help='异常值, 逗号分隔 (如 "1,-1")')
        parser.add_argument('--output', '-o', type=str, default='./output', help='输出文件夹路径')
        parser.add_argument('--K', type=int, default=5, help='离散化粒度 (默认: 5)')
        parser.add_argument('--lambda-ratio', type=float, default=0.05, help='核心概念选择比例 (默认: 0.05)')
        args = parser.parse_args()

        paths = [p.strip() for p in args.datasets.split(',') if p.strip()]
        anomaly_vals = [v.strip() for v in args.anomaly.split(',') if v.strip()]
        self.K = args.K
        self.lambda_ratio = args.lambda_ratio

        for fp in paths:
            if not os.path.exists(fp):
                print(f"错误: 文件不存在 - {fp}")
                continue
            df = pd.read_csv(fp)
            dataset_name = os.path.splitext(os.path.basename(fp))[0]
            self.dataset_configs.append({
                'file_path': fp, 'df_raw': df,
                'target_column': args.target, 'anomaly_values': anomaly_vals,
                'output_folder': args.output, 'dataset_name': dataset_name
            })

        if not self.dataset_configs:
            print("未配置任何有效数据集，程序退出。")
            return
        self._execute_pipeline()

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