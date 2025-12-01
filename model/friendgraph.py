import numpy as np
import csv
from collections import defaultdict
import numpy as np
import os
import scipy.sparse as sp
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

# 1) 读入朋友关系：friend_of[u] = [v1, v2, ...] （已映射为整数）
friend_of = defaultdict(list)
with open('/home/xrj/HST-POI/HST-POI-main/data/GW/gowalla_friend_list_mapped.csv', 'r', newline='') as f:
    for row in csv.reader(f):
        if not row:
            continue
        u = int(row[0])
        friends = list(map(int, row[1:]))   # 去掉空字符串
        friend_of[u] = friends


# 2) 计算地点总数（按 location_mapper 的最大索引 + 1）
location_mapper = np.load('/home/xrj/HST-POI/HST-POI-main/data/GW/location_mapper.npy', allow_pickle=True).item()
n_locations = len(location_mapper)           # 列数
user_mapper = np.load('/home/xrj/HST-POI/HST-POI-main/data/GW/user_mapper.npy', allow_pickle=True).item()
n_users = len(user_mapper)
print(n_locations)
print(n_users)

# 3) 预统计每个用户访问每个地点的次数
# user_loc_cnt[u][l] = 用户 u 访问地点 l 的次数
# 数据集路径
dataset_path = '/home/xrj/HST-POI/HST-POI-main/data/GW'
data_filename = 'train.csv'

# 加载映射字典
user2id_path = os.path.join(dataset_path, 'user_mapper.npy')
location2id_path = os.path.join(dataset_path, 'location_mapper.npy')
user2id = np.load(user2id_path, allow_pickle=True).item()
location2id = np.load(location2id_path, allow_pickle=True).item()

# 读取数据文件
data_file = os.path.join(dataset_path, data_filename)
with open(data_file, 'r') as f:
    lines = f.readlines()

# 解析数据
user_trajectories = {}
for line in lines:
    stay_points = line.strip().split(',')[1:]  # 获取除用户ID外的停留点数据
    user_id = line.strip().split(',')[0]  # 获取用户ID

    # 映射用户ID
    user_id_mapped = user2id[user_id]

    # 初始化用户轨迹
    if user_id_mapped not in user_trajectories:
        user_trajectories[user_id_mapped] = []

    for i in range(len(stay_points)):
        raw = stay_points[i].strip()
        # 1. 跳过真正的空串或不含@的字段（含<null>/<unset>/NaN转字符串后的'nan'）
        if not raw or '@' not in raw:
            continue
        # 2. 现在保证能拆出两项
        poi, timestamp = raw.split('@')
        # poi, timestamp = stay_points[i].split('@')  # 分割出地点和时间戳
        poi_id_mapped = location2id[poi]  # 映射POI ID
        user_trajectories[user_id_mapped].append((poi_id_mapped, int(timestamp)))

# 对每个用户的POI访问记录按时间戳排序
for user_id, records in user_trajectories.items():
    user_trajectories[user_id] = sorted(records, key=lambda x: x[1])

# 提取排序后的POI ID作为轨迹序列
user_trajectories_sequences = {user_id: [poi for poi, _ in records] for user_id, records in user_trajectories.items()}


user_loc_cnt = defaultdict(lambda: defaultdict(int))
for u, traj in user_trajectories_sequences.items():
    for l in traj:
        user_loc_cnt[u][l] += 1


# 4) 构造朋友访问频率矩阵
rows, cols, data = [], [], []

for u in range(n_users):
    friends = friend_of.get(u, [])
    if not friends:
        continue

    # 聚合朋友访问计数
    loc_cnt = defaultdict(int)
    total = 0
    for v in friends:
        total += sum(user_loc_cnt[v].values())  # 直接累加朋友的总访问量
        for l, cnt in user_loc_cnt[v].items():
            loc_cnt[l] += cnt

    if total == 0:
        continue

    # 填稀疏矩阵
    for l, cnt in loc_cnt.items():
        rows.append(u)
        cols.append(l)
        data.append(cnt / total)   # 相对频率


# 5) 保存结果
sparse_mat = sp.csr_matrix(
    (data, (rows, cols)),
    shape=(n_users, n_locations),
    dtype=np.float32
)

sp.save_npz('/home/xrj/HST-POI/HST-POI-main/data/GW/friend_visit_freq_sparse.npz', sparse_mat)
print('稀疏矩阵已保存：friend_visit_freq_sparse.npz')
print('shape =', sparse_mat.shape, 'nnz =', sparse_mat.nnz)