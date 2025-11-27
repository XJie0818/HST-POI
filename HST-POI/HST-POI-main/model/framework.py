import torch
from torch import nn
import math
from embedding import HSTEmbedding, PositionalEncoding
from Hypergraph import embedding_result, up_dim  # 如果是TC、MP数据集，使用这行
# from HypergraphGW import embedding_result, up_dim  # 如果是Gowalla数据集，使用这行
from GraphNet import GraphNet, PredictLayer
from visittime import NextVisitTime
from STFormer import TransEncoder, EnhancedTransformerStack


class HSTPOI(nn.Module):
    def __init__(self, config):
        super(HSTPOI, self).__init__()
        self.config = config
        self.base_dim = config.Embedding.base_dim
        self.use_graph_net = config.Model.use_graph_net
        self.up_num = up_dim

        self.embedding_layer = HSTEmbedding(config)

        if config.Encoder.encoder_type == 'trans':
            emb_dim = self.base_dim
            self.positional_encoding = PositionalEncoding(emb_dim=emb_dim)
            self.encoder = TransEncoder(config)

        if config.Encoder.encoder_type == 'LWtrans':
            emb_dim = self.base_dim
            self.positional_encoding = PositionalEncoding(emb_dim=emb_dim)
            self.encoder = EnhancedTransformerStack(config)

        fc_input_dim = self.base_dim

        if config.Model.at_type != 'none':
            self.at_net = NextVisitTime(config)
            fc_input_dim += self.base_dim

        # TC/MP
        fc_input_dim += self.base_dim

        if self.use_graph_net == 1:
            self.graph_net = GraphNet(input_dim=self.up_num, output_dim=self.base_dim)
            fc_input_dim += self.base_dim

        self.fc_layer = PredictLayer(input_dim=fc_input_dim,
                                     output_dim=config.Dataset.num_locations)
        self.out_dropout = nn.Dropout(0.1)

    def forward(self, batch_data):
        user_x = batch_data['user']
        loc_x = batch_data['location_x']
        hour_x = batch_data['hour']

        batch_size, sequence_length = loc_x.shape
        # print(batch_size,sequence_length)

        user_embedded, loc_embedded, timeslot_embedded = self.embedding_layer(batch_data)   # 生成嵌入向量
        time_embedded = timeslot_embedded[hour_x]

        lt_embedded = loc_embedded + time_embedded  # 将地点嵌入和时间嵌入相加得到lt_embedded

        if self.config.Encoder.encoder_type == 'trans':
            future_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(lt_embedded.device)
            future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
            encoder_out = self.encoder(self.positional_encoding(lt_embedded * math.sqrt(self.base_dim)),
                                       src_mask=future_mask)

        if self.config.Encoder.encoder_type == 'LWtrans':
            encoder_out = self.encoder(self.positional_encoding(lt_embedded * math.sqrt(self.base_dim)), loc_embedded, time_embedded)

        # combined = encoder_out
        combined = encoder_out + lt_embedded  # 将编码器输出与原始嵌入相加得到combined

        user_embedded = user_embedded[user_x]   # 根据用户ID user_x 从用户嵌入中提取对应的用户嵌入向量

        if self.config.Model.at_type != 'none':     # 注意力嵌入
            at_embedded = self.at_net(timeslot_embedded, batch_data)
            combined = torch.cat([combined, at_embedded], dim=-1)

        # TC/MP
        user_embedded = user_embedded.unsqueeze(1).repeat(1, sequence_length, 1)    # 用户嵌入
        combined = torch.cat([combined, user_embedded], dim=-1)

        if self.use_graph_net == 1:
            # loc_x = loc_x.to(embedding_result.device)
            user_x = user_x.to(embedding_result.device)
            pre_embedded = embedding_result[user_x]  # 提取对应的嵌入
            # pre_embedded = self.graph_net(pre_embedded).unsqueeze(1).repeat(1, sequence_length, 1)
            pre_embedded = pre_embedded.to(combined.device)
            pre_embedded = self.graph_net(pre_embedded).unsqueeze(1).repeat(1, sequence_length, 1)
            combined = torch.cat([combined, pre_embedded], dim=-1)

        out = self.fc_layer(combined.view(batch_size * sequence_length, combined.shape[2]))

        return out
