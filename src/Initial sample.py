import os
import pandas as pd

# ==========================================
# 1. 路径设置
# ==========================================
# 你的原始数据文件
input_csv_path = r"E:\Paper\lw6\python_code\Active_learnning\stage1\SSAP_all1.csv"

# 纯随机抽取的 200 个独立测试集的保存路径
output_csv_path = r"E:\Paper\lw6\python_code\Active_learnning\all\Test_Set_Fixed.csv"

# ==========================================
# 2. 数据读取与分离
# ==========================================
print("🚀 启动独立测试集抽样程序...")
print(f"📂 读取数据文件: {os.path.basename(input_csv_path)}")

# 读取整个数据集
df = pd.read_csv(input_csv_path)

# 自动获取最后一列的列名（目标 FoS 列）
target_col = df.columns[-1]

# 核心逻辑：分离已打标签（已测）和未打标签（未测）的数据
# dropna 保留非空值（已测样本），isna 保留空值（未测的搜索池样本）
labeled_df = df.dropna(subset=[target_col])
unlabeled_df = df[df[target_col].isna()]

print(f"   -> 发现已测试样本 (含FoS值): {len(labeled_df)} 个")
print(f"   -> 发现未测试样本 (无FoS值): {len(unlabeled_df)} 个")

# ==========================================
# 3. 纯随机盲抽 1000 个独立测试样本
# ==========================================
target_sample_size = 1000

# 安全检查：确保未测样本数量足够
if len(unlabeled_df) < target_sample_size:
    print(f"❌ 错误：未测试样本数量（{len(unlabeled_df)}）不足 {target_sample_size} 个！")
else:
    # 使用 sample 函数纯随机抽取，设置 random_state 保证可复现性（如果你不小心关了代码重跑，抽出来的还是这批）
    print(f"\n🎲 正在从 {len(unlabeled_df)} 个未测试样本中纯随机抽取 {target_sample_size} 个...")
    test_set_df = unlabeled_df.sample(n=target_sample_size, random_state=42)

    # 按照 FID (假设你的第一列是类似 ID 的标识符) 或索引排个序，方便查看
    if 'FID' in test_set_df.columns:
        test_set_df = test_set_df.sort_values(by='FID')
    else:
        test_set_df = test_set_df.sort_index()

    # ==========================================
    # 4. 导出保存
    # ==========================================
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    test_set_df.to_csv(output_csv_path, index=False)

    print("\n🎉 抽样完成！")
    print(f"✅ 独立测试集已成功保存至:\n📂 {output_csv_path}")
    print("\n⚠️ 接下来请将这 200 个样本放入 SSAP 中计算 FoS 值。")
    print("⚠️ 算完后，这个文件将作为你所有迭代轮次的唯一评估标尺，绝不能放入训练集中！")