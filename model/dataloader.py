import datetime
import os
import random
import gensim
import numpy as np
import torch
from gensim import models
from torch.utils.data import Dataset
from tqdm import tqdm


def _set_seed(seed=256):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 让 gensim 内部也固定
    gensim.utils.RANDOM = np.random.RandomState(seed)
    # 若用多线程 / 多 worker，再锁哈希
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_weekday_hour(timestamp):
    dt = datetime.datetime.fromtimestamp(int(timestamp) // 1000)
    weekday = dt.weekday()
    hour = dt.hour
    return weekday, hour


class HSTDataset(Dataset):
    # 如果需要完成随机丢弃签到的鲁棒性实验，请修改此处
    # def __init__(self, config, dataset_path, device, load_mode, drop_ratio=0.0, seed=42):
    def __init__(self, config, dataset_path, device, load_mode):
        _set_seed(256)
        self.config = config
        self.device = device
        self.load_mode = load_mode
        self.dataset_path = dataset_path
        self.user2id = np.load(os.path.join(dataset_path, 'user_mapper.npy'), allow_pickle=True).item()
        self.location2id = np.load(os.path.join(dataset_path, 'location_mapper.npy'), allow_pickle=True).item()

        if load_mode == 'test':
            self.data = self.load_npy_file(os.path.join(dataset_path, f'{load_mode}.npy'))
        else:
            if not os.path.exists(os.path.join(dataset_path, f'{load_mode}.npy')):
                self.preprocess_data()
                # 生成TC与MP的数据
                # self.generate_data(load_mode='train')
                # self.generate_data(load_mode='test')
                # 生成GW的数据
                self.generate_gw_data(load_mode='train')
                self.generate_gw_data(load_mode='test')
            self.data = self.load_npy_file(os.path.join(dataset_path, f'{load_mode}.npy'))

            # 鲁棒性实验中，随机根据比例丢弃训练数据
            # if drop_ratio > 0 and load_mode == 'train':
            #     rng = np.random.default_rng(seed)
            #     n_drop = int(len(self.data) * drop_ratio)
            #     drop_idx = rng.choice(len(self.data), n_drop, replace=False)
            #     self.data = [d for i, d in enumerate(self.data) if i not in drop_idx]
            #     print(f'[Robustness] dropped {n_drop}/{n_drop + len(self.data)} training samples')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        return data

    def preprocess_data(self):
        trans_time_individual = []
        occur_time_individual = np.zeros(shape=(len(self.user2id), 24), dtype=np.float32)
        user_loc_matrix = np.zeros((len(self.user2id), len(self.location2id)))

        diff_data = []
        with open(os.path.join(self.dataset_path, f'train.csv'), 'r', encoding='utf8') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc=f'Preprocess data'):
                trans_matrix_time = np.ones((24, 24))
                stay_points = line.strip().split(',')[1:]
                user = line.strip().split(',')[0]
                for i in range(len(stay_points) - 1):
                    # 处理空字符串
                    cur = stay_points[i].strip()
                    nxt = stay_points[i + 1].strip()
                    if '@' not in cur or '@' not in nxt:  # 跳过 <null>/<unset>/空串
                        continue
                    location, timestamp = cur.split('@')
                    next_location, next_timestamp = nxt.split('@')
                    weekday, hour = extract_weekday_hour(timestamp)
                    next_weekday, next_hour = extract_weekday_hour(next_timestamp)
                    diff_data.append(abs(next_hour - hour))
                    i, j = hour, next_hour
                    trans_matrix_time[i, j] += 1
                    occur_time_individual[self.user2id[user]][hour] += 1
                    user_loc_matrix[self.user2id[user], self.location2id[location]] += 1
                    if i == len(stay_points) - 2:
                        occur_time_individual[self.user2id[user]][next_hour] += 1
                        user_loc_matrix[self.user2id[user], self.location2id[next_location]] += 1

                time_row_sums = trans_matrix_time.sum(axis=1)
                trans_matrix_time = trans_matrix_time / time_row_sums[:, np.newaxis]
                trans_time_individual.append(trans_matrix_time)

        trans_time_individual = np.array(trans_time_individual)

        num_users, num_locations = user_loc_matrix.shape
        dictionary = gensim.corpora.Dictionary([[str(i)] for i in range(num_locations)])
        corpus = []
        for user in user_loc_matrix:
            user_doc = [str(loc) for loc, count in enumerate(user) for _ in range(int(count))]
            corpus.append(dictionary.doc2bow(user_doc))

        np.save(os.path.join(self.dataset_path, f'prob_matrix_time_individual.npy'),
                np.array(trans_time_individual))
        np.save(os.path.join(self.dataset_path, f'occur_time_individual.npy'),
                np.array(occur_time_individual))

    def generate_data(self, load_mode):
        occur_time_individual = np.load(os.path.join(self.dataset_path, f'occur_time_individual.npy'),
                                        allow_pickle=True)
        res = []
        with open(os.path.join(self.dataset_path, f'{load_mode}.csv'), 'r', encoding='utf8') as file:
            lines = file.readlines()
            for line_i, line in enumerate(tqdm(lines, desc=f'Initial {load_mode} data')):
                user = line.strip().split(',')[0]
                occur_time_user = occur_time_individual[self.user2id[user]]
                stay_points = line.strip().split(',')[1:]
                sequence_count, left = divmod(len(stay_points), self.config.Dataset.sequence_length)
                assert sequence_count > 0, f"{user}'s does not have enough data."
                sequence_count -= 1 if left == 0 else 0
                for i in range(sequence_count):
                    split_start = i * self.config.Dataset.sequence_length
                    split_end = (i + 1) * self.config.Dataset.sequence_length
                    location_x = [self.location2id[item.split('@')[0]] for item in stay_points[split_start:split_end]]
                    timestamp_x = [item.split('@')[1] for item in stay_points[split_start:split_end]]
                    location_y = [self.location2id[item.split('@')[0]] for item in
                                  stay_points[split_start + 1:split_end + 1]]
                    timestamp_y = [item.split('@')[1] for item in stay_points[split_start + 1:split_end + 1]]
                    timeslot_y = []
                    hour_x = []
                    hour_mask = []
                    for item in timestamp_x:
                        weekday, hour = extract_weekday_hour(item)
                        hour_x.append(hour)
                        mask = np.zeros(24, dtype=np.int32)
                        mask[occur_time_user == 0] = 1
                        if mask.sum() == 24:
                            exit()
                        hour_mask.append(mask)
                    for item in timestamp_y:
                        weekday, hour = extract_weekday_hour(item)
                        timeslot_y.append(hour)
                    res.append(
                        {
                            'user': self.user2id[user],
                            'location_x': location_x,
                            'hour': hour_x,
                            'hour_mask': np.array(hour_mask),
                            'location_y': location_y,
                            'timeslot_y': timeslot_y,
                        }
                    )

        np.save(os.path.join(self.dataset_path, f'{load_mode}.npy'), res)

    def generate_gw_data(self, load_mode):
        occur_time_individual = np.load(os.path.join(self.dataset_path, f'occur_time_individual.npy'),
                                        allow_pickle=True)
        res = []

        with open(os.path.join(self.dataset_path, f'{load_mode}.csv'), 'r', encoding='utf8') as file:
            lines = file.readlines()
            for line_i, line in enumerate(tqdm(lines, desc=f'Initial {load_mode} data')):
                user = line.strip().split(',')[0]
                occur_time_user = occur_time_individual[self.user2id[user]]
                stay_points = line.strip().split(',')[1:]

                seq_len = self.config.Dataset.sequence_length
                sequence_count, left = divmod(len(stay_points), seq_len)
                assert sequence_count > 0, f"{user} does not have enough data."
                if left == 0:
                    sequence_count -= 1

                for i in range(sequence_count):
                    split_start = i * seq_len
                    split_end = (i + 1) * seq_len
                    # 安全拆分
                    location_x, timestamp_x = [], []
                    for item in stay_points[split_start:split_end]:
                        item = str(item).strip()
                        if '@' not in item:
                            continue
                        parts = item.split('@')
                        if len(parts) != 2:
                            continue
                        loc, ts = parts
                        if loc in self.location2id:  # 防未知 POI
                            location_x.append(self.location2id[loc])
                            timestamp_x.append(ts)

                    location_y, timestamp_y = [], []
                    for item in stay_points[split_start + 1:split_end + 1]:
                        item = str(item).strip()
                        if '@' not in item:
                            continue
                        parts = item.split('@')
                        if len(parts) != 2:
                            continue
                        loc, ts = parts
                        if loc in self.location2id:
                            location_y.append(self.location2id[loc])
                            timestamp_y.append(ts)

                    # 若过滤后长度不足，直接丢弃该片段
                    if len(location_x) != seq_len or len(location_y) != seq_len:
                        continue

                    hour_x, hour_mask, timeslot_y = [], [], []
                    for ts in timestamp_x:
                        weekday, hour = extract_weekday_hour(ts)
                        hour_x.append(hour)
                        mask = np.zeros(24, dtype=np.int32)
                        mask[occur_time_user == 0] = 1
                        if mask.sum() == 24:
                            exit()
                        hour_mask.append(mask)

                    for ts in timestamp_y:
                        weekday, hour = extract_weekday_hour(ts)
                        timeslot_y.append(hour)

                    res.append({
                        'user': self.user2id[user],
                        'location_x': location_x,
                        'hour': hour_x,
                        'hour_mask': np.array(hour_mask),
                        'location_y': location_y,
                        'timeslot_y': timeslot_y,
                    })

        np.save(os.path.join(self.dataset_path, f'{load_mode}.npy'), res)

    def load_npy_file(self, save_path):
        loaded_data = np.load(save_path, allow_pickle=True)
        prob_matrix_time_individual = np.load(
            os.path.join(self.dataset_path, f'prob_matrix_time_individual.npy'),
            allow_pickle=True)
        for data in loaded_data:
            user_idx = data['user']
            data['prob_matrix_time_individual'] = prob_matrix_time_individual[user_idx]
        return loaded_data

