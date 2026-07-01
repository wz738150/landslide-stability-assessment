import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 0. Global Chart Settings (Top-Tier Journal Style)
# ==========================================
# 使用顶刊常用的 Times New Roman 字体
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
    "axes.spines.top": True,
    "axes.spines.right": True
})

# ==========================================
# 1. Data Loading & Splitting (Labeled vs Unlabeled Pool)
# ==========================================
file_path = r"E:\Paper\lw6\python_code\Active_learnning\SSAP_Features.csv"
output_csv_path = r"E:\Paper\lw6\python_code\Active_learnning\AL_Round1_Selected_50.csv"
output_img_path = r"E:\Paper\lw6\python_code\Active_learnning\Figure_2_Phase_Space_Selection.png"

print("Reading data...")
df = pd.read_csv(file_path)
target_col = df.columns[-1]

# 假设前三列包含 FID, X, Y，特征从第4列(索引3)开始到倒数第二列
# 将数据分为已标记（用于训练）和未标记（用于预测和挑选）的池
labeled_df = df.dropna(subset=[target_col]).copy()
unlabeled_df = df[df[target_col].isna()].copy()

if unlabeled_df.empty:
    raise ValueError("未标记数据池为空！请确保 CSV 中包含目标列(FoS)为空的样本池。")

# 提取训练特征和标签
X_train = labeled_df.iloc[:, 3:-1].values
y_train = labeled_df[target_col].values

# 提取候选池特征
X_pool = unlabeled_df.iloc[:, 3:-1].values

# ==========================================
# 2. Model Training & Uncertainty Quantification
# ==========================================
print("Training Random Forest model on labeled data...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("Calculating Epistemic Uncertainty for the unlabeled pool...")
# 获取森林中每一棵树的预测结果
# preds_trees 形状: (n_estimators, n_samples)
preds_trees = np.stack([tree.predict(X_pool) for tree in rf_model.estimators_])

# 计算均值和标准差（不确定性度量）
pool_preds_mean = np.mean(preds_trees, axis=0)
pool_uncertainty = np.std(preds_trees, axis=0)

# 将预测结果和不确定性加回 unlabeled_df
unlabeled_df['Predicted_FoS'] = pool_preds_mean
unlabeled_df['Uncertainty_Std'] = pool_uncertainty

# ==========================================
# 3. Active Learning Selection (Query Strategy)
# ==========================================
print("Selecting top 50 samples with highest uncertainty...")
n_select = 50
# 按不确定性降序排列，取前50个
selected_df = unlabeled_df.sort_values(by='Uncertainty_Std', ascending=False).head(n_select)

# 导出指定的列：FID, X, Y
# 注意：请确保原CSV中存在这三列且名称大小写匹配，若为小写请相应修改为 'fid', 'x', 'y'
selected_df[['FID', 'X', 'Y']].to_csv(output_csv_path, index=False)
print(f"已成功挑选 {n_select} 个样本并保存至: {output_csv_path}")

# ==========================================
# 4. Visualization: Rank-Decay & Feature Trend Plot (Zero Scatter)
# ==========================================
fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)

# 1. 核心数据重构：按不确定性从大到小严格排序
df_sorted = unlabeled_df.sort_values(by='Uncertainty_Std', ascending=False).reset_index(drop=True)

# 构造X轴：排序名次 (Rank 1 to N)
ranks = np.arange(1, len(df_sorted) + 1)
uncertainty = df_sorted['Uncertainty_Std'].values
fos = df_sorted['Predicted_FoS'].values

# 2. 绘制左轴 (不确定性瀑布面积图)
# 选中部分 (Top 50) 使用深红色填充
ax1.fill_between(ranks[:n_select], uncertainty[:n_select],
                 color='#C44E52', alpha=0.85, label=f'Targeted Batch (Top {n_select})')
# 剩余候选池使用高级浅灰色填充
ax1.fill_between(ranks[n_select-1:], uncertainty[n_select-1:],
                 color='#E0E0E0', alpha=0.6, label=f'Remaining Pool (N={len(df_sorted)-n_select})')
# 绘制不确定性衰减的边界实线
ax1.plot(ranks, uncertainty, color='#333333', linewidth=1.5, zorder=3)

# 绘制垂直截断线
ax1.axvline(x=n_select, color='#8B0000', linestyle='--', linewidth=1.5, zorder=4)
ax1.text(n_select + 50, uncertainty[0] * 0.9, 'Selection Cut-off',
         color='#8B0000', fontsize=11, fontweight='bold', ha='left', va='top')

ax1.set_xlabel('Sample Rank (Sorted by Epistemic Uncertainty)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Epistemic Uncertainty', fontweight='bold', fontsize=12, color='#8B0000')
ax1.tick_params(axis='y', labelcolor='#8B0000')

# 限制X轴范围，可以放大头部区域（比如只显示前 1500 名，让断崖更清晰）
# 如果想看全貌，可以注释掉下面这行
ax1.set_xlim(0, max(1000, len(df_sorted) * 0.2)) # 这里设定只看前20%以凸显头部，可调

# 3. 绘制右轴 (预测FoS平滑趋势线)
ax2 = ax1.twinx()

# 为了避免折线过于毛刺，使用滑动平均 (Rolling Mean) 提取宏观趋势
window_size = max(10, int(len(df_sorted) * 0.01))
fos_smooth = pd.Series(fos).rolling(window=window_size, min_periods=1, center=True).mean()

ax2.plot(ranks, fos_smooth, color='#4C72B0', linewidth=2.5, linestyle='-',
         label=f'Predicted FoS Trend (Moving Avg, w={window_size})')

# 添加地质临界区带状阴影 (FoS < 1.2)
critical_threshold = 1.2
ax2.axhspan(0, critical_threshold, color='#4C72B0', alpha=0.1, zorder=1)
ax2.axhline(critical_threshold, color='#3A5380', linestyle=':', linewidth=1.5)

# 在右侧标出临界区文字
ax2.text(ax1.get_xlim()[1] * 0.98, critical_threshold - 0.1,
         'Critical Zone (FoS < 1.2)',
         color='#3A5380', fontsize=11, fontweight='bold', ha='right', va='top')

ax2.set_ylabel('Predicted Factor of Safety (Trend)', fontweight='bold', fontsize=12, color='#4C72B0')
ax2.tick_params(axis='y', labelcolor='#4C72B0')
# 假设FoS不会小于0，设置合理下限
ax2.set_ylim(bottom=0)

# 4. 图例整合与布局美化
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right',
           frameon=True, edgecolor='black', fancybox=False)

plt.title('Active Learning Query Profile: Uncertainty Decay vs. Safety Factor Trend',
          fontweight='bold', fontsize=13, pad=15)

plt.tight_layout()
plt.savefig(output_img_path, bbox_inches='tight')
plt.show()

print("已成功生成并保存无点化（排序衰减）高级图表。")