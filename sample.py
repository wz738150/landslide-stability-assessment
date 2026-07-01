import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 模型评估与预处理
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_curve)

# 六大核心代理模型 (全部转换为 Classifier 二分类器)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局出图设置 (符合顶刊标准)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

sns.set_theme(style="ticks", rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.spines.top": False,
    "axes.spines.right": False
})

# ==========================================
# 1. 定义数据路径字典
# ==========================================
# 独立的静态测试集 (N=200)
TEST_SET_PATH = r"E:\Paper\lw6\python_code\Active_learnning\all\Test_Set_Fixed.csv"

# 迭代训练集
file_paths = {
    'Round 0': r"J:\Paper\lw6\python_code\data\SSAP_Features.csv",
}

# ==========================================
# 2. 定义多分类模型矩阵 (内置类别平衡参数)
# ==========================================
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42,
                                            n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=8, tree_method='hist', random_state=42),
    # XGBoost权重在循环内动态计算
    'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=8, class_weight='balanced',
                               random_state=42, verbose=-1, n_jobs=-1),
    'CatBoost': CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, auto_class_weights='Balanced',
                                   silent=True, random_seed=42),
    'SVC': SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True),
    'ANN': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True, random_state=42)
}

stage_results = []
THRESHOLD = 1.2  # 物理失稳临界阈值

# ==========================================
# 3. 加载全局静态测试集
# ==========================================
print("📥 Loading Independent Hold-out Test Set (Binary Classification)...")
try:
    test_df = pd.read_csv(TEST_SET_PATH)
    target_col = test_df.columns[-1]
    test_df = test_df.dropna(subset=[target_col]).copy()

    X_test_raw = test_df.iloc[:, 3:-1].values
    y_test_raw = test_df[target_col].values

    # 【核心逻辑】：FoS < 1.2 为危险(1)，FoS >= 1.2 为安全(0)
    # 机器学习中必须将我们要找的“灾害/异常”设定为正类(1)
    y_test_bin = (y_test_raw < THRESHOLD).astype(int)

    print(f"   -> 测试集加载成功！共包含独立样本: {len(test_df)} 个")
    print(f"   -> 物理状态分布: 危险(1)={np.sum(y_test_bin == 1)}, 安全(0)={np.sum(y_test_bin == 0)}\n")

    # 冻结特征空间坐标系
    scaler = StandardScaler()
    scaler.fit(X_test_raw)
    X_test = scaler.transform(X_test_raw)

except FileNotFoundError:
    print(f"\n❌ 错误: 找不到独立的测试集文件 {TEST_SET_PATH}")
    exit()

# ==========================================
# 4. 动态演化评估 (包含 Youden's J 动态阈值)
# ==========================================
print("🚀 Starting Active Learning Evaluation with Dynamic Thresholding...\n")

for stage_name, path in file_paths.items():
    file_name = os.path.basename(path)
    print(f"[{stage_name}] Loading Training Data: {file_name}")

    train_df = pd.read_csv(path)
    train_df = train_df.dropna(subset=[target_col]).copy()

    X_train_raw = train_df.iloc[:, 3:-1].values
    y_train_raw = train_df[target_col].values

    # 训练集标签二值化：危险为1，安全为0
    y_train_bin = (y_train_raw < THRESHOLD).astype(int)

    negative_count = np.sum(y_train_bin == 1)  # 危险点数量
    positive_count = np.sum(y_train_bin == 0)  # 安全点数量
    print(f"   -> 当前 AL 池分布: 危险点(1)={negative_count}, 安全点(0)={positive_count}")

    X_train = scaler.transform(X_train_raw)

    # 动态计算 XGBoost 的不平衡比重
    scale_pos_w = positive_count / (negative_count + 1e-5)
    models['XGBoost'].set_params(scale_pos_weight=scale_pos_w)

    for model_name, model in models.items():
        # 1. 训练二分类模型
        model.fit(X_train, y_train_bin)

        # 2. 预测测试集和训练集的【概率】
        y_pred_prob_test = model.predict_proba(X_test)[:, 1]
        y_pred_prob_train = model.predict_proba(X_train)[:, 1]

        # 3. 【核心突破】：利用训练集 ROC 曲线寻找最佳物理切分阈值 (Youden's J Index)
        # 废弃死板的 0.5，让模型根据数据分布自己决定警报线
        fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_train_bin, y_pred_prob_train)
        optimal_idx = np.argmax(tpr_tr - fpr_tr)
        best_thresh = thresholds_tr[optimal_idx]

        # 如果最佳阈值过于极端，做一个安全保护限制 (保证在合理区间)
        best_thresh = np.clip(best_thresh, 0.2, 0.8)

        # 4. 使用寻找出的最佳阈值，对测试集进行硬分类切分
        y_pred_bin_test = (y_pred_prob_test >= best_thresh).astype(int)

        # 5. 计算指标
        auc = roc_auc_score(y_test_bin, y_pred_prob_test)
        prec = precision_score(y_test_bin, y_pred_bin_test, zero_division=0)
        rec = recall_score(y_test_bin, y_pred_bin_test, zero_division=0)
        f1 = f1_score(y_test_bin, y_pred_bin_test, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin_test).ravel()

        stage_results.append({
            'Stage': stage_name, 'Model': model_name,
            'AUC': auc, 'Best_Thresh': best_thresh,
            'F1': f1, 'Precision': prec, 'Recall': rec,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
        })
    print("-" * 60)

results_df = pd.DataFrame(stage_results)

# ==========================================
# 5. 打印学术级汇总表格 (按 AUC 寻优)
# ==========================================
print("\n" + "=" * 115)
print("📊 FINAL RESULTS: METRICS ON HOLD-OUT TEST SET (Optimal Threshold Applied)")
print("=" * 115)

best_rows = []
for stage in results_df['Stage'].unique():
    stage_data = results_df[results_df['Stage'] == stage]
    # 按照空间预测的金标准 AUC 挑选每轮的最强 MVP 模型
    best_row = stage_data.loc[stage_data['AUC'].idxmax()]
    best_rows.append(best_row)

rep_df = pd.DataFrame(best_rows)

header = f"{'Stage':<8} | {'MVP Model':<14} | {'AUC-ROC':<8} | {'Opt_Thresh':<10} | {'F1-Score':<8} | {'Prec':<6} | {'Rec':<6} | {'TP/FP/TN/FN'}"
print(header)
print("-" * 115)

for idx, row in rep_df.iterrows():
    cm_str = f"{int(row['TP'])}/{int(row['FP'])}/{int(row['TN'])}/{int(row['FN'])}"
    line = (
        f"{row['Stage']:<8} | {row['Model']:<14} | {row['AUC']:<8.3f} | {row['Best_Thresh']:<10.3f} | {row['F1']:<8.3f} | "
        f"{row['Precision']:<6.2f} | {row['Recall']:<6.2f} | {cm_str}")
    print(line)

print("=" * 115 + "\n")
print("注: 废弃了默认的 0.5 概率阈值，采用 Youden's J Statistic 在训练集自适应寻找最佳易发性截断点。")
print(f"    正类(Positive/1) = 危险边坡 (FoS < {THRESHOLD})，负类(Negative/0) = 安全边坡 (FoS >= {THRESHOLD})。")

# ==========================================
# 6. 绘制高水平学术收敛曲线 (AUC vs F1)
# ==========================================
print("🎨 Rendering Journal-Quality Convergence Plot (AUC & F1)...")
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

stages = rep_df['Stage'].values
auc_scores = rep_df['AUC'].values
f1_scores = rep_df['F1'].values
mvp_models = rep_df['Model'].values

# 左Y轴：AUC-ROC (衡量模型全局判别能力)
color1 = '#2b5c8f'
ax1.set_xlabel('Active Learning Iterations', fontweight='bold', fontsize=13)
ax1.set_ylabel('Global AUC-ROC Score', color=color1, fontweight='bold', fontsize=13)
line1 = ax1.plot(stages, auc_scores, marker='o', markersize=9, color=color1, linewidth=3, label='Optimal AUC')
ax1.tick_params(axis='y', labelcolor=color1)

y_min, y_max = min(auc_scores), max(auc_scores)
ax1.set_ylim(max(0.4, y_min - 0.1), min(1.0, y_max + 0.05))

for i, txt in enumerate(mvp_models):
    ax1.annotate(txt, (stages[i], auc_scores[i]), textcoords="offset points", xytext=(0, 15),
                 ha='center', fontsize=10, color='#1c3c5e', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color1, alpha=0.9))

# 右Y轴：F1-Score (衡量临界失稳状态的捕捉能力)
ax2 = ax1.twinx()
color2 = '#c44e52'
ax2.set_ylabel('Critical State F1-Score (Youden Optimized)', color=color2, fontweight='bold', fontsize=13)
line2 = ax2.plot(stages, f1_scores, marker='s', markersize=9, color=color2, linewidth=3, linestyle='--',
                 label='Optimal F1')
ax2.tick_params(axis='y', labelcolor=color2)

y2_min, y2_max = min(f1_scores), max(f1_scores)
ax2.set_ylim(max(0.1, y2_min - 0.1), min(1.0, y2_max + 0.15))

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', frameon=True, edgecolor='black', fontsize=11)

ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
plt.title(f'Binary Classification Capability Evolution (Dynamic Thresholding)', fontweight='bold', fontsize=15, pad=20)
plt.tight_layout()

# 如需保存图片，可取消下行注释
# plt.savefig('Learning_Curve_Holdout_Binary.png', bbox_inches='tight')

plt.show()
print("✅ 运行完毕。")