import pandas as pd
import numpy as np
import rasterio
import fiona
from shapely.geometry import shape

# --- 路径配置 ---
shp_path = r"E:\Paper\lw6\dataset\landslide\slope_unit.shp"
dem_path = r"E:\Paper\lw6\dataset\factor\DEM.tif"
aspect_path = r"E:\Paper\lw6\dataset\factor\Aspect.tif"
lit_path = r"E:\Paper\lw6\dataset\factor\Lit.tif"  # 岩性数据
output_csv = r"E:\Paper\lw6\dataset\landslide\SSAP_Profiles_Final.csv"

# --- 参数设置 ---
SAMPLE_SPACING = 30.0
BUFFER_RATIO = 1.2


def get_dynamic_points(centroid_x, centroid_y, angle_deg, max_dim, spacing):
    math_angle = np.radians(90 - angle_deg)
    half_len = (max_dim * BUFFER_RATIO) / 2
    num_steps = int(half_len * 2 / spacing)
    if num_steps < 2: num_steps = 2

    x_coords = np.linspace(centroid_x - np.cos(math_angle) * half_len,
                           centroid_x + np.cos(math_angle) * half_len, num_steps)
    y_coords = np.linspace(centroid_y - np.sin(math_angle) * half_len,
                           centroid_y + np.sin(math_angle) * half_len, num_steps)
    return zip(x_coords, y_coords)


def main():
    results = []

    with rasterio.open(dem_path) as dem_ds, \
            rasterio.open(aspect_path) as asp_ds, \
            rasterio.open(lit_path) as lit_ds, \
            fiona.open(shp_path) as shp_src:

        dem_data = dem_ds.read(1)
        asp_data = asp_ds.read(1)
        lit_data = lit_ds.read(1)

        nodata_dem = dem_ds.nodata
        print(f"开始处理 {len(shp_src)} 个单元...")

        for feature in shp_src:
            fid = feature['properties'].get('FID', feature['id'])
            geom_obj = shape(feature['geometry'])
            centroid = geom_obj.centroid
            minx, miny, maxx, maxy = geom_obj.bounds
            max_dim = max(maxx - minx, maxy - miny)

            # 获取主坡向
            r_a, c_a = asp_ds.index(centroid.x, centroid.y)
            try:
                m_aspect = asp_data[r_a, c_a] if (0 <= r_a < asp_ds.height and 0 <= c_a < asp_ds.width) else 0
                if m_aspect < 0 or m_aspect > 360: m_aspect = 0
            except:
                m_aspect = 0

            directions = {"Main": m_aspect, "Plus45": (m_aspect + 45) % 360, "Minus45": (m_aspect - 45) % 360}

            for l_type, angle in directions.items():
                points = get_dynamic_points(centroid.x, centroid.y, angle, max_dim, SAMPLE_SPACING)

                # 记录当前剖面的有效点
                line_points = []
                for px, py in points:
                    r, c = dem_ds.index(px, py)
                    if 0 <= r < dem_ds.height and 0 <= c < dem_ds.width:
                        z = dem_data[r, c]
                        # 过滤无效值
                        if z == nodata_dem or z == 65535 or z > 9000 or z < -500:
                            continue

                        # 提取岩性
                        r_l, c_l = lit_ds.index(px, py)
                        try:
                            lit_val = lit_data[r_l, c_l] if (
                                        0 <= r_l < lit_ds.height and 0 <= c_l < lit_ds.width) else 1
                        except:
                            lit_val = 1

                        line_points.append([px, py, z, lit_val])

                # --- 关键修复：重置 Distance ---
                if line_points:
                    first_x, first_y = line_points[0][0], line_points[0][1]
                    for pt in line_points:
                        # 计算当前点相对于本剖面第一个有效点的距离
                        curr_dist = np.sqrt((pt[0] - first_x) ** 2 + (pt[1] - first_y) ** 2)
                        results.append({
                            "FID": fid,
                            "LineType": l_type,
                            "X": round(pt[0], 3),
                            "Y": round(pt[1], 3),
                            "Z": round(float(pt[2]), 2),
                            "Distance": round(curr_dist, 3),
                            "Lithology": int(pt[3])
                        })

    # 导出
    df = pd.DataFrame(results)
    # 按照要求的顺序排队：FID, LineType, X, Y, Z, Distance, Lithology
    df = df[["FID", "LineType", "X", "Y", "Z", "Distance", "Lithology"]]
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"完成！每一条线的 Distance 现在都从 0 开始。结果保存在: {output_csv}")


if __name__ == "__main__":
    main()