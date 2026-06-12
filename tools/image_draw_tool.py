import pandas as pd
import numpy as np
import matplotlib

# 强制使用标准后端，解决部分 IDE 内置绘图兼容性问题
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import os
import warnings

# 忽略 sklearn 中因分母为0产生的警告
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 🎨 绘图字体配置：防止乱码，强制使用英文标准字体
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
# 优先使用 Arial，兼容各系统
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================

def main():
    print("=" * 50)
    print("🔍 Anomaly Detection Metrics Plotting Tool")
    print("Tip: Enter 'end' or 'q' to stop input")
    print("=" * 50)

    algo_data = {}

    # 1. Loop to read CSV files
    while True:
        filepath = input("\n📂 Enter CSV file path: ").strip()
        if filepath.lower() in ['end', 'q', 'quit', '']:
            break
        if not os.path.exists(filepath):
            print("❌ File not found. Please check path.")
            continue

        algo_name = input(f"🏷️ Enter Algorithm Name for '{os.path.basename(filepath)}': ").strip()
        if not algo_name:
            algo_name = f"Algorithm_{len(algo_data) + 1}"

        try:
            # Read CSV: Col 2 is Score, Col 4 is Actual Result
            df = pd.read_csv(filepath, header=0)
            if df.shape[1] < 4:
                print("❌ CSV format error: Requires at least 4 columns.")
                continue

            raw_scores = df.iloc[:, 1].astype(float).values
            raw_actuals = df.iloc[:, 3].astype(float).values

            # ==========================================
            # 🔥 [Logic] Sort data by Score Descending
            # ==========================================
            # argsort returns ascending indices, negate to sort descending
            sorted_indices = np.argsort(-raw_scores)

            # Reorder scores and labels accordingly
            scores = raw_scores[sorted_indices]
            actuals = raw_actuals[sorted_indices]

            # Binarize actuals: 1 for anomaly (>0), 0 for normal
            actuals = (actuals > 0).astype(int)
            # ==========================================

            algo_data[algo_name] = (scores, actuals)
            print(f"✅ Loaded '{algo_name}'. Sorted descending. Samples: {len(scores)}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")

    if not algo_data:
        print("\n⚠️ No data loaded. Exiting.")
        return

    # 2. Set Save Folder
    print("\n" + "-" * 50)
    folder_input = input("📁 Enter Save Folder (e.g., D:\\results\\, Enter for current dir): ").strip()

    if not folder_input:
        save_folder = os.getcwd()
        print(f"👉 Saving to current directory: {save_folder}")
    else:
        save_folder = os.path.normpath(folder_input)
        if os.path.isfile(save_folder):
            save_folder = os.path.dirname(save_folder)

        if not os.path.exists(save_folder):
            try:
                os.makedirs(save_folder)
                print(f"📁 Created directory: {save_folder}")
            except Exception as e:
                print(f"❌ Cannot create directory: {e}")
                save_folder = os.getcwd()

    # 3. Calculate Metrics
    print("\n⏳ Calculating Precision, Recall, F1 for each k...")

    metrics_dict = {algo: {'precision': [], 'recall': [], 'f1': []} for algo in algo_data}

    for algo_name, (scores, actuals) in algo_data.items():
        n_samples = len(scores)
        for k in range(n_samples):
            # Threshold is the score at rank k (0-indexed)
            # Since sorted descending, k=0 is highest score
            threshold = scores[k]

            # Prediction: Score >= Threshold is Anomaly (1)
            preds = (scores >= threshold).astype(int)

            p = precision_score(actuals, preds, zero_division=0)
            r = recall_score(actuals, preds, zero_division=0)
            f1 = f1_score(actuals, preds, zero_division=0)

            metrics_dict[algo_name]['precision'].append(p)
            metrics_dict[algo_name]['recall'].append(r)
            metrics_dict[algo_name]['f1'].append(f1)

    # ==========================================
    # 🔥 Choose Plot Mode: Metrics or ROC
    # ==========================================
    print("\n" + "-" * 50)
    print("📊 Select Plot Mode:")
    print("  1 - Precision/Recall/F1 Curves (default)")
    print("  2 - ROC Curves")
    print("  3 - Both")
    mode = input("👉 Enter choice (1/2/3, default=1): ").strip()
    if mode not in ['1', '2', '3']:
        mode = '1'

    colors = plt.cm.tab10(np.linspace(0, 1, len(algo_data)))

    # 4A. Generate Precision/Recall/F1 Plots
    if mode in ['1', '3']:
        metric_configs = [
            {'key': 'precision', 'name': 'Precision', 'file': 'precision_curve.png'},
            {'key': 'recall', 'name': 'Recall', 'file': 'recall_curve.png'},
            {'key': 'f1', 'name': 'F1-Score', 'file': 'f1_curve.png'}
        ]

        for config in metric_configs:
            key = config['key']
            name = config['name']
            filename = config['file']

            plt.figure(figsize=(10, 6))

            for i, algo_name in enumerate(algo_data.keys()):
                data_list = metrics_dict[algo_name][key]
                # X-axis: k from 1 to N
                k_values = range(1, len(data_list) + 1)

                plt.plot(k_values, data_list,
                         label=algo_name, color=colors[i % len(colors)],
                         linewidth=2, marker='o', markersize=4, markevery=max(1, len(data_list) // 40))

            plt.xlabel('k (Threshold Index)', fontsize=12, fontweight='bold')
            plt.ylabel(name, fontsize=12, fontweight='bold')
            plt.title(f'{name} vs k', fontsize=14, fontweight='bold', pad=15)
            plt.legend(fontsize=10, loc='best', frameon=True, shadow=True)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.xlim(0, max(len(metrics_dict[a][key]) for a in algo_data))
            plt.ylim(0, 1.05)
            plt.tight_layout()

            save_path = os.path.join(save_folder, filename)
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"🖼️ Saved: {save_path}")
            except Exception as e:
                print(f"❌ Save failed: {e}")

            plt.close()

    # 4B. Generate ROC Curve Plot
    if mode in ['2', '3']:
        print("\n📈 Generating ROC Curves...")

        plt.figure(figsize=(10, 8))

        for i, algo_name in enumerate(algo_data.keys()):
            scores, actuals = algo_data[algo_name]

            # sklearn roc_curve expects: positive class = 1, scores where higher = more likely positive
            # Our data: actuals already binarized (1=anomaly), scores sorted descending (higher=more anomalous)
            fpr, tpr, thresholds = roc_curve(actuals, scores)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr,
                     label=f'{algo_name} (AUC = {roc_auc:.4f})',
                     color=colors[i % len(colors)],
                     linewidth=2)

        # Diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')

        plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=15)
        plt.legend(fontsize=9, loc='lower right', frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()

        roc_save_path = os.path.join(save_folder, 'roc_curve.png')
        try:
            plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
            print(f"🖼️ Saved: {roc_save_path}")
        except Exception as e:
            print(f"❌ Save failed: {e}")

        plt.close()

    print("\n✅ Process complete!")


if __name__ == "__main__":
    main()