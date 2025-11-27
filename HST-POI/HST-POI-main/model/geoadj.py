import numpy as np
import os
import scipy.sparse as sp
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from math import radians, cos, sin, asin, sqrt

# 经纬度数据集路径
dataset_path = '/home/xrj/HST-POI/HST-POI-main/data/GW'
coords_file = os.path.join(dataset_path, 'gowalla_final_poi_coords_mapped.csv')


def load_poi_coords(coords_file):
    poi_coords = pd.read_csv(coords_file)
    session_dict = {row['spot']: (row['lon'], row['lat']) for _, row in poi_coords.iterrows()}
    return session_dict


# 加载 POI 坐标，生成 session_dict
session_dict = load_poi_coords(coords_file)

# 获取 POI 的数量
num_pois = len(session_dict)
print(num_pois)


def haversine_distance(lon1, lat1, lon2, lat2):
    """Haversine distance"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球半径（单位：公里）
    return c * r


def gen_sparse_geo_H_poi(session_dict, distance_threshold=20):  # 可更改地理距离阈值
    # 获取 POI 的总数
    num_pois = len(session_dict)

    # 初始化邻接矩阵
    H_geo = np.zeros((num_pois, num_pois))

    # 获取所有 POI ID
    poi_ids = list(session_dict.keys())

    # 遍历所有 POI 对，计算 Haversine 距离
    for i in range(num_pois):
        for j in range(i + 1, num_pois):  # 只计算上三角矩阵，避免重复计算
            lon1, lat1 = session_dict[poi_ids[i]]
            lon2, lat2 = session_dict[poi_ids[j]]
            distance = haversine_distance(lon1, lat1, lon2, lat2)
            if distance < distance_threshold:  # 如果距离小于阈值
                H_geo[i, j] = 1
                H_geo[j, i] = 1  # 无向图，对称设置

    # 将邻接矩阵转换为稀疏矩阵
    H_geo_sparse = sp.csr_matrix(H_geo)
    return H_geo_sparse


# 生成地理超图的邻接矩阵
H_geo_sparse = gen_sparse_geo_H_poi(session_dict, distance_threshold=20)
print(H_geo_sparse.shape)

# 保存稀疏矩阵
H_geo_path = os.path.join(dataset_path, 'H_geo_sparse20km.npz')
sp.save_npz(H_geo_path, H_geo_sparse)

print(f"地理超图邻接矩阵已保存到 {H_geo_path}")
