import numpy as np
import os
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# 数据集路径
dataset_path = '/home/xrj/HST-POI/HST-POI-main/data/MP'
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
print(len(user_trajectories_sequences))


def gen_sparse_H_user_freq(sessions_dict, num_pois, num_users):
    rows, cols, data = [], [], []

    for userID, seq in sessions_dict.items():
        # 计算当前用户的POI访问频次
        poi_counts = {}
        for poi in seq:
            poi_counts[poi] = poi_counts.get(poi, 0) + 1

        # 归一化：计算每个POI的访问概率 P(poi|user) = count(poi) / len(seq)
        seq_len = len(seq)
        for poi, count in poi_counts.items():
            rows.append(poi)
            cols.append(userID)
            data.append(count / seq_len)  # 访问频率

    H = sp.csr_matrix((data, (rows, cols)), shape=(num_pois, num_users))
    return H


def gen_sparse_directed_H_poi_with_prob(users_trajs_dict, num_pois):
    # 统计原始转移次数
    transfer_counts = np.zeros(shape=(num_pois, num_pois))
    for userID, traj in users_trajs_dict.items():
        for src_idx in range(len(traj) - 1):
            src_poi = traj[src_idx]
            tar_poi = traj[src_idx + 1]
            transfer_counts[src_poi, tar_poi] += 1

    #  归一化为转移概率
    row_sums = transfer_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除以0（无转移的POI行保持全0）
    transfer_probs = transfer_counts / row_sums

    H = sp.csr_matrix(transfer_probs)

    return H


def csr_matrix_drop_edge(csr_adj_matrix, keep_rate):
    if keep_rate == 1.0:
        return csr_adj_matrix

    coo = csr_adj_matrix.tocoo()
    row = coo.row
    col = coo.col
    edgeNum = row.shape[0]

    # generate edge mask
    mask = np.floor(np.random.rand(edgeNum) + keep_rate).astype(np.bool_)

    # get new values and indices
    new_row = row[mask]
    new_col = col[mask]
    new_edgeNum = new_row.shape[0]
    new_values = np.ones(new_edgeNum, dtype=float)

    drop_adj_matrix = sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)

    return drop_adj_matrix


def get_hyper_deg(incidence_matrix):

    eps = 1e-8
    rowsum = np.array(incidence_matrix.sum(1))
    d_inv = np.power(rowsum + eps, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    return d_mat_inv


num_pois = len(location2id)
num_users = len(user2id)
print(num_pois)
print(num_users)

H_pu = gen_sparse_H_user_freq(user_trajectories_sequences, num_pois, num_users)  # 用户超图 (L,U)
H_pu = csr_matrix_drop_edge(H_pu, 1.0)
Deg_H_pu = get_hyper_deg(H_pu)
HG_pu = Deg_H_pu * H_pu

H_up = H_pu.T  # 用户超图逆(U,L)
Deg_H_up = get_hyper_deg(H_up)
HG_up = Deg_H_up * H_up

H_poi_src = gen_sparse_directed_H_poi_with_prob(user_trajectories_sequences, num_pois)  # 移动超图 (L,L)
H_poi_src = csr_matrix_drop_edge(H_poi_src, 0.8)
Deg_H_poi_src = get_hyper_deg(H_poi_src)
HG_poi_src = Deg_H_poi_src * H_poi_src

H_poi_tar = H_poi_src.T
Deg_H_poi_tar = get_hyper_deg(H_poi_tar)
HG_poi_tar = Deg_H_poi_tar * H_poi_tar


up_dim = 200
poi_emb = nn.Embedding(num_pois, up_dim)  # POI嵌入
nn.init.xavier_uniform_(poi_emb.weight)
poi_emb_matrix = poi_emb.weight.data
poi_emb_tensor = torch.tensor(poi_emb_matrix, dtype=torch.float32)
# poi_emb = poi_emb.to(device)

# 二套门控参数（全局）
w_gate_up = torch.empty(up_dim, up_dim)
b_gate_up = torch.empty(1, up_dim)
w_gate_trans = torch.empty(up_dim, up_dim)
b_gate_trans = torch.empty(1, up_dim)

# def xavier_normal_(t):
#     fan_in, fan_out = t.shape[0], t.shape[1]
#     std = (2.0 / (fan_in + fan_out)) ** 0.5
#     with torch.no_grad():
#         t.normal_(0, std)

# 权重和偏置都使用Xavier正态初始化
nn.init.xavier_normal_(w_gate_up)
nn.init.xavier_normal_(b_gate_up)      # 偏置也用Xavier
nn.init.xavier_normal_(w_gate_trans)
nn.init.xavier_normal_(b_gate_trans)   # 偏置也用Xavier

raw_emb = poi_emb_tensor

# 用户超图
HG_pu_tensor = torch.tensor(HG_pu.toarray(), dtype=torch.float32)
HG_up_tensor = torch.tensor(HG_up.toarray(), dtype=torch.float32)

gate_up = torch.sigmoid(raw_emb @ w_gate_up + b_gate_up)
poi_emb_tensor = raw_emb * gate_up  # 直接替换原 poi_emb_tensor
final_pre_emb = [poi_emb_tensor]
dropout = nn.Dropout(0.1)

# 一层卷积
msg_poi_agg = torch.sparse.mm(HG_up_tensor, poi_emb_tensor)  # 超图卷积 POI->轨迹超边
pre_emb = torch.sparse.mm(HG_pu_tensor, msg_poi_agg)  # 轨迹超边->POI
pre_emb = pre_emb + final_pre_emb[-1]  # 残差连接
# pre_emb = F.relu(pre_emb)  # 激活函数
pre_emb = dropout(pre_emb)
final_pre_emb.append(pre_emb)

# 二层卷积
msg_poi_agg1 = torch.sparse.mm(HG_up_tensor, pre_emb)
pre_emb = torch.sparse.mm(HG_pu_tensor, msg_poi_agg1)
pre_emb = pre_emb + final_pre_emb[-1]
pre_emb = dropout(pre_emb)
final_pre_emb.append(pre_emb)
# final_pre_emb = torch.mean(torch.stack(final_pre_emb), dim=0)  # 取均值
# print(final_pre_emb.shape)

# 三层卷积
# msg_poi_agg2 = torch.sparse.mm(HG_up_tensor, pre_emb)
# pre_emb = torch.sparse.mm(HG_pu_tensor, msg_poi_agg2)
# pre_emb = pre_emb + final_pre_emb[-1]
# pre_emb = dropout(pre_emb)
# final_pre_emb.append(pre_emb)

# 四层卷积
# msg_poi_agg3 = torch.sparse.mm(HG_up_tensor, pre_emb)
# pre_emb = torch.sparse.mm(HG_pu_tensor, msg_poi_agg3)
# pre_emb = pre_emb + final_pre_emb[-1]
# pre_emb = dropout(pre_emb)
# final_pre_emb.append(pre_emb)

final_pre_emb = torch.mean(torch.stack(final_pre_emb), dim=0)  # 取均值
final_user_emb = torch.sparse.mm(HG_up_tensor, final_pre_emb)
# print(final_user_emb.shape)


# 移动超图
HG_poi_src_tensor = torch.tensor(HG_poi_src.toarray(), dtype=torch.float32)
HG_poi_tar_tensor = torch.tensor(HG_poi_tar.toarray(), dtype=torch.float32)

gate_trans = torch.sigmoid(raw_emb @ w_gate_trans + b_gate_trans)
poi_emb1_tensor = raw_emb * gate_trans
# poi_emb1_tensor = raw_emb
final_pre_emb1 = [poi_emb1_tensor]

# 一层卷积
msg_tar = torch.sparse.mm(HG_poi_tar_tensor, poi_emb1_tensor)  # 超图卷积,源节点->目标节点
msg_src = torch.sparse.mm(HG_poi_src_tensor, msg_tar)  # 目标节点->源节点
pre_emb1 = msg_src + final_pre_emb1[-1]  # 残差连接
# pre_emb1 = F.relu(pre_emb1)
pre_emb1 = dropout(pre_emb1)
final_pre_emb1.append(pre_emb1)

# 二层卷积
msg_tar1 = torch.sparse.mm(HG_poi_tar_tensor, pre_emb1)
msg_src1 = torch.sparse.mm(HG_poi_src_tensor, msg_tar1)
pre_emb1 = msg_src1 + final_pre_emb1[-1]
pre_emb1 = dropout(pre_emb1)
final_pre_emb1.append(pre_emb1)

# 三层卷积
# msg_tar2 = torch.sparse.mm(HG_poi_tar_tensor, pre_emb1)
# msg_src2 = torch.sparse.mm(HG_poi_src_tensor, msg_tar2)
# pre_emb1 = msg_src2 + final_pre_emb1[-1]
# pre_emb1 = dropout(pre_emb1)
# final_pre_emb1.append(pre_emb1)

# 四层卷积
# msg_tar3 = torch.sparse.mm(HG_poi_tar_tensor, pre_emb1)
# msg_src3 = torch.sparse.mm(HG_poi_src_tensor, msg_tar3)
# pre_emb1 = msg_src3 + final_pre_emb1[-1]
# pre_emb1 = dropout(pre_emb1)
# final_pre_emb1.append(pre_emb1)

final_pre_emb1 = torch.mean(torch.stack(final_pre_emb1), dim=0)  # 取均值
final_user_emb1 = torch.sparse.mm(HG_up_tensor, final_pre_emb1)
# print(final_user_emb1.shape)

hg_emb_all = final_user_emb
trans_emb_all = final_user_emb1

hg_emb_all = F.normalize(hg_emb_all, p=2, dim=1)
trans_emb_all = F.normalize(trans_emb_all, p=2, dim=1)

Hyper_gate = nn.Sequential(nn.Linear(up_dim, 1), nn.Sigmoid())  # 门控机制
Trans_gate = nn.Sequential(nn.Linear(up_dim, 1), nn.Sigmoid())


embedding_result = Hyper_gate(hg_emb_all) * hg_emb_all + Trans_gate(trans_emb_all) * trans_emb_all  # 自适应融合

print(embedding_result)
print(embedding_result.shape)


