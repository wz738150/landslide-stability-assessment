import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict

# ==========================================
# 0. Global Chart Settings (Top-Tier Journal Style)
# ==========================================
# 使用顶刊常用的 Times New Roman 字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2      # 加粗边框线
plt.rcParams['xtick.direction'] = 'in'    # 刻度线向内
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

# 使用干净的 ticks 主题，去掉多余网格线，保留四周的边框
sns.set_theme(style="ticks", rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.spines.top": True,
    "axes.spines.right": True
})

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
file_path = r"E:\Paper\lw6\python_code\Active_learnning\stage2\SSAP_Features1.csv"

print("Reading data...")
df = pd.read_csv(file_path)
target_col = df.columns[-1]

labeled_df = df.dropna(subset=[target_col]).copy()
X = labeled_df.iloc[:, 3:-1].values
y = labeled_df[target_col].values

# ==========================================
# 2. Model Training & Prediction (Cross-Validation)
# ==========================================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Performing cross-validation predictions...")
y_pred = cross_val_predict(rf_model, X, y, cv=kf)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
residuals = y_pred - y

# ==========================================
# 3. Visualization (Journal Quality)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

# ------------------------------------------
# Figure 1a: True vs Predicted FoS
# ------------------------------------------
ax1 = axes[0]
# 绘制散点与回归带，使用沉稳的学术蓝(#4C72B0)和红(#C44E52)
sns.regplot(x=y, y=y_pred, ax=ax1,
            scatter_kws={'alpha': 0.7, 'edgecolor': 'k', 'linewidth': 0.5, 's': 45, 'color': '#4C72B0'},
            line_kws={'color': '#C44E52', 'label': 'Regression line (95% CI)', 'linewidth': 2})

# 绘制 1:1 基准线，稍微延伸出散点范围
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
margin = (max_val - min_val) * 0.05
ax1.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
         'k--', lw=1.5, label='1:1 line')

ax1.set_xlabel('True Factor of Safety (FoS)', fontweight='bold')
ax1.set_ylabel('Predicted Factor of Safety (FoS)', fontweight='bold')
ax1.set_title('(a)', loc='left', fontweight='bold', fontsize=14)
ax1.legend(loc='upper left', frameon=False) # 图例无边框，更显高级

# 【修复重叠】将文本框移动到右下角
text_str = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
props = dict(boxstyle='square,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.8)
ax1.text(0.95, 0.05, text_str, transform=ax1.transAxes, fontsize=11,
         verticalalignment='bottom', horizontalalignment='right', bbox=props)

# ------------------------------------------
# Figure 1b: Residual Distribution
# ------------------------------------------
ax2 = axes[1]
# 绘制直方图和核密度图
sns.histplot(residuals, kde=True, bins=15, color='#4C72B0', ax=ax2, stat='density',
             edgecolor='black', linewidth=0.5, alpha=0.6)

res_mean = np.mean(residuals)
res_std = np.std(residuals)

# 使用深灰色标示标准差，红色标示均值
ax2.axvline(res_mean, color='#C44E52', linestyle='--', lw=2, label=f'Mean: {res_mean:.3f}')
ax2.axvline(res_mean + res_std, color='#555555', linestyle=':', lw=1.5, label=f'+1 SD: {res_mean + res_std:.3f}')
ax2.axvline(res_mean - res_std, color='#555555', linestyle=':', lw=1.5, label=f'-1 SD: {res_mean - res_std:.3f}')

ax2.set_xlabel('Prediction Residual (Predicted - True FoS)', fontweight='bold')
ax2.set_ylabel('Density', fontweight='bold')
ax2.set_title('(b)', loc='left', fontweight='bold', fontsize=14)
# 【已修改】图例位置从 upper right 改为 upper left
ax2.legend(loc='upper left', frameon=False)

# 调整布局并保存图像 (bbox_inches='tight' 保证标签不被裁切)
plt.tight_layout()
plt.savefig(r'E:\Paper\lw6\python_code\Active_learnning\Figure_1_Baseline_Performance_Journal.png', bbox_inches='tight')
plt.show()

print("图表已按照英文期刊标准生成并保存。")