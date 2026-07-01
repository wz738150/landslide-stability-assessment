import pandas as pd

# 文件路径（直接用你的路径）
path_active = r"E:\Paper\lw6\python_code\Active_learnning\stage3\AL_Round3_Dynamic_Optimal.csv"
path_ssap = r"E:\Paper\lw6\python_code\Active_learnning\stage4\SSAP_Features3.csv"
# 读取两个CSV
df_active = pd.read_csv(path_active, encoding="utf-8")
df_ssap = pd.read_csv(path_ssap, encoding="utf-8")

# 获取要更新的列名（最后一列）
target_col = df_ssap.columns[-1]

# 创建FID映射
fid_to_value = dict(zip(df_active["FID"], df_active[target_col]))

# 关键：只填充空值，不覆盖原来已有的150个数据
df_ssap[target_col] = df_ssap[target_col].fillna(df_ssap["FID"].map(fid_to_value))

# 保存
df_ssap.to_csv(path_ssap, index=False, encoding="utf-8")

# ============= 输出结果信息 =============
print("✅ 操作完成！")
print(f"✅ 目标列：{target_col}")
print(f"✅ 本次新增填充行数：{df_ssap['FID'].isin(df_active['FID']).sum()} 行")
print(f"✅ 总有效数据行数：{df_ssap[target_col].notna().sum()} 行")
print(f"✅ 文件已保存到：{path_ssap}")