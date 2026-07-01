import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 0. Global Chart Settings (Top-Tier Journal Style)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

sns.set_theme(style="ticks", rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.spines.top": False,
    "axes.spines.right": False
})

# ==========================================
# 1. Data Loading & Splitting
# ==========================================
file_path = r"E:\Paper\lw6\python_code\Active_learnning\stage2\SSAP_Features1.csv"
output_csv_path = r"E:\Paper\lw6\python_code\Active_learnning\stage2\AL_Round2_Selected_50_Optimal.csv"
output_img_profile = r"E:\Paper\lw6\python_code\Active_learnning\stage2\Figure_3b_PIUS_Profile.png"

print("Reading data...")
df = pd.read_csv(file_path)
target_col = df.columns[-1]

labeled_df = df.dropna(subset=[target_col]).copy()
unlabeled_df = df[df[target_col].isna()].copy()

print(f"==== Data Status ====")
print(f"Current Labeled (Training) Pool Size: {len(labeled_df)}")
print(f"Remaining Unlabeled Pool Size: {len(unlabeled_df)}")
print(f"=====================")

X_train = labeled_df.iloc[:, 3:-1].values
y_train = labeled_df[target_col].values
X_pool = unlabeled_df.iloc[:, 3:-1].values

# ==========================================
# 2. Model Training & Base Predictions
# ==========================================
print("\nTraining Random Forest model on updated labeled data (Round 2)...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("Calculating Epistemic Uncertainty for the remaining unlabeled pool...")
preds_trees = np.stack([tree.predict(X_pool) for tree in rf_model.estimators_])

pool_preds_mean = np.mean(preds_trees, axis=0)
pool_uncertainty = np.std(preds_trees, axis=0)

unlabeled_df['Predicted_FoS'] = pool_preds_mean
unlabeled_df['Uncertainty_Std'] = pool_uncertainty

# ==========================================
# 3. Final Extraction with Fixed Optimal Beta (1.4)
# ==========================================
optimal_beta = 1.4
critical_fos = 1.2
n_select = 50

print(f"\nExtracting final 50 samples using Fixed Optimal Beta = {optimal_beta}...")

# 【消除量纲影响】：将不确定性和FoS偏差进行 Min-Max 归一化 (0~1)
u_min, u_max = unlabeled_df['Uncertainty_Std'].min(), unlabeled_df['Uncertainty_Std'].max()
p_dist = np.abs(unlabeled_df['Predicted_FoS'] - critical_fos)
p_min, p_max = p_dist.min(), p_dist.max()

unlabeled_df['U_norm'] = (unlabeled_df['Uncertainty_Std'] - u_min) / (u_max - u_min + 1e-9)
unlabeled_df['P_norm'] = (p_dist - p_min) / (p_max - p_min + 1e-9)

# 计算归一化 PIUS 综合得分
unlabeled_df['Physics_Score'] = unlabeled_df['U_norm'] - optimal_beta * unlabeled_df['P_norm']

# 提取排名前 50 的样本并保存
selected_df_optimal = unlabeled_df.sort_values(by='Physics_Score', ascending=False).head(n_select)
selected_df_optimal[['FID', 'X', 'Y']].to_csv(output_csv_path, index=False)
print(f"已成功输出最优 50 样本文件至: {output_csv_path}")

# ==========================================
# 4. Plot: Rank-Decay & Feature Trend Plot
# ==========================================
fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)

# 按物理约束综合得分降序排列用于绘图
df_sorted = unlabeled_df.sort_values(by='Physics_Score', ascending=False).reset_index(drop=True)

ranks = np.arange(1, len(df_sorted) + 1)
acq_scores = df_sorted['Physics_Score'].values
fos = df_sorted['Predicted_FoS'].values

# 左轴 (PIUS 综合得分面积图)
ax1.fill_between(ranks[:n_select], acq_scores[:n_select], acq_scores.min(),
                 color='#C44E52', alpha=0.85, label=f'Targeted Batch (Top {n_select})')
ax1.fill_between(ranks[n_select - 1:], acq_scores[n_select - 1:], acq_scores.min(),
                 color='#E0E0E0', alpha=0.6, label=f'Remaining Pool (N={len(df_sorted) - n_select})')
ax1.plot(ranks, acq_scores, color='#333333', linewidth=1.5, zorder=3)

ax1.axvline(x=n_select, color='#8B0000', linestyle='--', linewidth=1.5, zorder=4)
ax1.text(n_select + 50, acq_scores[0] * 0.9, 'Selection Cut-off', color='#8B0000', fontsize=11, fontweight='bold')

ax1.set_xlabel('Sample Rank (Sorted by Normalized PIUS Score)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Acquisition Score (PIUS)', fontweight='bold', fontsize=12, color='#8B0000')
ax1.tick_params(axis='y', labelcolor='#8B0000')
ax1.set_ylim(bottom=acq_scores.min() * 1.05 if acq_scores.min() < 0 else 0)
ax1.set_xlim(0, max(1000, len(df_sorted) * 0.2))

# 右轴 (预测FoS平滑趋势线)
ax2 = ax1.twinx()
window_size = max(10, int(len(df_sorted) * 0.01))
fos_smooth = pd.Series(fos).rolling(window=window_size, min_periods=1, center=True).mean()

ax2.plot(ranks, fos_smooth, color='#4C72B0', linewidth=2.5, linestyle='-', label=f'Predicted FoS Trend (Moving Avg)')
ax2.axhspan(0, critical_fos, color='#4C72B0', alpha=0.1, zorder=1)
ax2.axhline(critical_fos, color='#3A5380', linestyle=':', linewidth=1.5)
ax2.text(ax1.get_xlim()[1] * 0.98, critical_fos - 0.1, f'Critical Zone (FoS < {critical_fos})', color='#3A5380',
         fontsize=11, fontweight='bold', ha='right', va='top')

ax2.set_ylabel('Predicted Factor of Safety (Trend)', fontweight='bold', fontsize=12, color='#4C72B0')
ax2.tick_params(axis='y', labelcolor='#4C72B0')
ax2.set_ylim(bottom=0)

# 图例整合
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True, edgecolor='black')

plt.title(f'Physics-Informed Active Learning Profile (Optimal Normalized $\\beta$={optimal_beta})', fontweight='bold',
          fontsize=13, pad=15)

# 机制说明框
text_str = (f"Query Strategy: Normalized PIUS\n"
            f"Score = $\sigma_{{norm}}$ - {optimal_beta} $\\times$ | $\mu - {critical_fos} |_{{norm}}$\n"
            f"Current Train Size: {len(labeled_df)}")
props = dict(boxstyle='square,pad=0.6', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.8)
ax1.text(0.03, 0.96, text_str, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props, zorder=5)

plt.tight_layout()
plt.savefig(output_img_profile, bbox_inches='tight')
plt.show()

print(f"已保存最优参数下的相空间分布图至: {output_img_profile}")