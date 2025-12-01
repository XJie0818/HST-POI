import os
import csv
import numpy as np

# 1. 读映射
dataset_path = '/home/xrj/HST-POI/HST-POI-main/data/GW'
loc_mapper_path = os.path.join(dataset_path, 'location_mapper.npy')
if not os.path.exists(loc_mapper_path):
    raise FileNotFoundError(f'{loc_mapper_path} 不存在，请先运行 get_mapper 生成。')
location2id = np.load(loc_mapper_path, allow_pickle=True).item()   # {str: int}

# 2. 路径
src_csv = os.path.join(dataset_path, 'gowalla_final_poi_coords_aligned.csv')      # 原始三列：poi_str,lon,lat
dst_csv = os.path.join(dataset_path, 'gowalla_final_poi_coords_mapped.csv')

# 3. 转换并写出
with open(src_csv, 'r', newline='', encoding='utf-8') as fin, \
     open(dst_csv, 'w', newline='', encoding='utf-8') as fout:

    reader = csv.reader(fin)
    writer = csv.writer(fout)

    # 写标题
    writer.writerow(['spot', 'lon', 'lat'])

    for row in reader:
        if not row:                # 跳过空行
            continue
        raw_loc, lon, lat = row[0], row[1], row[2]
        if raw_loc not in location2id:   # 防御：映射里找不到就跳过
            continue
        mapped_id = location2id[raw_loc]
        writer.writerow([mapped_id, lon, lat])

print('转换完成，已保存到', dst_csv)