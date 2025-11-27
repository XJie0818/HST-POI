import numpy as np
import csv
import os

# 1. 加载映射
mapper_path = '/home/xrj/HST-POI/HST-POI-main/data/GW/user_mapper.npy'                    # 生成的映射文件
if not os.path.exists(mapper_path):
    raise FileNotFoundError(f'{mapper_path} 不存在，请先运行 get_mapper 生成。')

user2id = np.load(mapper_path, allow_pickle=True).item()   # {str: int}

# 2. 转换并写出
src_csv = '/home/xrj/HST-POI/HST-POI-main/data/GW/gowalla_friend_list.csv'
dst_csv = '/home/xrj/HST-POI/HST-POI-main/data/GW/gowalla_friend_list_mapped.csv'

with open(src_csv, 'r', newline='', encoding='utf-8') as fin, \
     open(dst_csv, 'w', newline='', encoding='utf-8') as fout:

    reader = csv.reader(fin)
    writer = csv.writer(fout)

    for row in reader:
        if not row:               # 跳过空行
            continue
        # 原文件每行格式：uid, friend1, friend2, ...
        mapped = [user2id.get(uid, -1) for uid in row]
        writer.writerow(mapped)

print('转换完成，已保存到', dst_csv)