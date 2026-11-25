import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 读取 CSV 文件
csv_file_path = r'C:\Users\LiuQian\PycharmProjects\PythonProject1\ChinaCrimeDatas.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(csv_file_path)

# 读取 SHP 文件
shp_file_path = 'roi/hangzhou.shp'  # 替换为你的SHP文件路径
gdf = gpd.read_file(shp_file_path)

# 创建一个空的 GeoDataFrame 来存储符合条件的点
filtered_data = []

# 遍历数据
for index, row in data.iterrows():
    # 提取经纬度
    latitude = row['latitude']
    longitude = row['longitude']

    # 创建点对象
    point = Point(longitude, latitude)

    # 检查点是否在 SHP 文件定义的地理范围内
    for geometry in gdf.geometry:
        if geometry.contains(point):
            # 如果在范围内，将该行数据添加到结果列表
            filtered_data.append(row)
            break  # 匹配成功后退出内层循环

# 将结果列表转换为 DataFrame
filtered_df = pd.DataFrame(filtered_data)

# 保存结果到新的 CSV 文件
output_csv_path = 'filtered_data.csv'  # 输出文件路径
filtered_df.to_csv(output_csv_path, index=False)

print(f"符合条件的数据已保存到 {output_csv_path}")
