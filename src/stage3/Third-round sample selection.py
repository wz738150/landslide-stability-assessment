import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict

# 引入万能集成器 (用于统一输出不确定性)
from sklearn.ensemble import BaggingRegressor

# 六大核心模型
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. Global Settings
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# ==========================================
# 1. Data Loading, Capping & Standardization
# ==========================================
file_path = r"E:\Paper\lw6\python_code\Active_learnning\stage3\SSAP_Features2.csv"
output_csv_path = r"E:\Paper\lw6\python_code\Active_learnning\stage3\AL_Round3_Dynamic_Optimal.csv"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

df = pd.read_csv(file_path)
target_col = df.columns[-1]

labeled_df = df.dropna(subset=[target_col]).copy()
unlabeled_df = df[df[target_col].isna()].copy()

X_train_raw = labeled_df.iloc[:, 3:-1].values
y_train_raw = labeled_df[target_col].values  # 原始含极大值的 FoS
X_pool_raw = unlabeled_df.iloc[:, 3:-1].values

# 【核心铁律】：目标变量物理截断 (Capped at 5.0)
FOS_CAP = 5.0
y_train = np.clip(y_train_raw, a_min=None, a_max=FOS_CAP)
print(f"✅ Target (FoS) successfully capped at {FOS_CAP} for surrogate training.")

# 【必须保留】特征标准化，确保 SVR 和 ANN 公平竞技
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_pool = scaler.transform(X_pool_raw)

print(f"\n==== Stage 1 (Dynamic Evaluation & PIUS Sampling) ====")
print(f"Labeled Pool: {len(labeled_df)} | Unlabeled Pool: {len(unlabeled_df)}")

# ==========================================
# 2. Dynamic Model Evaluation (择优阶段)
# ==========================================
print("\n[Step 1] Evaluating 6 paradigm models via 5-Fold CV on Capped Data...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=8, tree_method='hist', random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=8, random_state=42, verbose=-1, n_jobs=-1),
    'CatBoost': CatBoostRegressor(iterations=150, learning_rate=0.05, depth=6, silent=True, random_seed=42),
    'SVR': SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'),
}

best_model_name = ""
best_r2 = -float('inf')
best_model_instance = None

print(f"{'Model':<15} | {'Global R2':<10} | {'RMSE':<8} | {'RMSE (FoS<1.2)':<15}")
print("-" * 55)

for name, model in models.items():
    y_pred = cross_val_predict(model, X_train, y_train, cv=kf)
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))

    critical_mask = y_train < 1.2
    rmse_c = np.sqrt(mean_squared_error(y_train[critical_mask], y_pred[critical_mask])) if np.sum(
        critical_mask) > 0 else np.nan

    print(f"{name:<15} | {r2:<10.3f} | {rmse:<8.3f} | {rmse_c:<15.3f}")

    # 根据 R2 选出当前轮次的“王牌模型”
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model_instance = model

print("-" * 55)
print(f"🎯 胜出模型: {best_model_name} (R2 = {best_r2:.3f}) 将作为本轮样本挑选的引擎！\n")

# ==========================================
# 3. Universal Uncertainty Quantification (统一量化阶段)
# ==========================================
print(f"[Step 2] Building Universal Deep Ensembles for {best_model_name}...")

# 核心魔法：使用 BaggingRegressor 克隆 5 个胜出模型，强行获取其认知不确定性
ensemble_model = BaggingRegressor(estimator=best_model_instance, n_estimators=5, random_state=42, n_jobs=-1)
ensemble_model.fit(X_train, y_train)

# 获取 5 个克隆体对未知样本的独立预测
ensemble_preds = np.stack([m.predict(X_pool) for m in ensemble_model.estimators_])

pool_preds_mean = np.mean(ensemble_preds, axis=0)
pool_uncertainty = np.std(ensemble_preds, axis=0)

unlabeled_df['Predicted_FoS'] = pool_preds_mean
unlabeled_df['Uncertainty_Std'] = pool_uncertainty

# ==========================================
# 4. PIUS 物理约束靶向采样 (采样阶段)
# ==========================================
# 你刚才算出的最优 beta 是 1.3，这里沿用
optimal_beta = 1.6
critical_fos = 1.2
n_select = 50

print(f"\n[Step 3] Extracting top {n_select} samples using Dynamic PIUS (Beta = {optimal_beta})...")

u_min, u_max = unlabeled_df['Uncertainty_Std'].min(), unlabeled_df['Uncertainty_Std'].max()
p_dist = np.abs(unlabeled_df['Predicted_FoS'] - critical_fos)
p_min, p_max = p_dist.min(), p_dist.max()

unlabeled_df['U_norm'] = (unlabeled_df['Uncertainty_Std'] - u_min) / (u_max - u_min + 1e-9)
unlabeled_df['P_norm'] = (p_dist - p_min) / (p_max - p_min + 1e-9)

# 计算物理综合得分
unlabeled_df['Physics_Score'] = unlabeled_df['U_norm'] - optimal_beta * unlabeled_df['P_norm']

# 保存最优样本
selected_df_optimal = unlabeled_df.sort_values(by='Physics_Score', ascending=False).head(n_select)
selected_df_optimal[['FID', 'X', 'Y']].to_csv(output_csv_path, index=False)
print(f"✅ 成功输出第三轮挑选的 {n_select} 个待计算样本至:\n📂 {output_csv_path}")