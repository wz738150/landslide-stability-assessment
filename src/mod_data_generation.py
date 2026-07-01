import os
import glob

# --- 1. 路径配置 ---
# 存放 .dat 和 .geo 文件的文件夹
data_dir = r"E:\Paper\lw6\dataset\ssap\ssap_dat"


def main():
    # 查找所有的 .dat 文件
    dat_files = glob.glob(os.path.join(data_dir, "*.dat"))

    if not dat_files:
        print(f"❌ 错误：在 {data_dir} 中未找到 .dat 文件。")
        return

    print(f"🚀 开始处理，共检测到 {len(dat_files)} 个任务...")

    success_count = 0

    for dat_path in dat_files:
        # 获取文件名（例如: 0_Main.dat）
        dat_filename = os.path.basename(dat_path)
        # 获取不带后缀的文件名（例如: 0_Main）
        base_name = os.path.splitext(dat_filename)[0]

        # 定义对应的 .geo 文件名和要生成的 .mod 文件路径
        geo_filename = f"{base_name}.geo"
        mod_path = os.path.join(data_dir, f"{base_name}.mod")

        # 检查对应的 .geo 是否存在，保证索引有效
        if not os.path.exists(os.path.join(data_dir, geo_filename)):
            print(f"⚠️ 跳过 {base_name}: 未找到对应的 .geo 文件")
            continue

        # --- 2. 构建 .mod 文件内容 ---
        # 根据你提供的格式：
        # 第一行是固定标志位（带前置空格）
        # 第二行是 dat 文件名
        # 第三行是 geo 文件名
        mod_content = [
            "    1    0    0    0    0    0    0    0    0    0",
            dat_filename,
            geo_filename
        ]

        # --- 3. 写入文件 ---
        try:
            with open(mod_path, 'w', encoding='utf-8') as f:
                # 写入每一行并换行
                f.write("\n".join(mod_content) + "\n")
            success_count += 1
        except Exception as e:
            print(f"❌ 写入 {base_name}.mod 失败: {e}")

        # 每隔 1000 个打印一次进度
        if success_count % 1000 == 0:
            print(f"已完成 {success_count} 个文件...")

    print("-" * 50)
    print(f"✅ 处理完成！成功生成 {success_count} 个 .mod 文件。")


if __name__ == "__main__":
    main()