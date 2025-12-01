import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.nn import init


class LocalWindowAttention(nn.Module):  # 窗口注意力
    def __init__(self, config):
        super(LocalWindowAttention, self).__init__()
        self.config = config
        embed_dim = config.Embedding.base_dim
        num_heads = config.Embedding.num_heads
        window_size = config.Embedding.window_size
        self.mha = MultiheadAttention(embed_dim, num_heads)
        self.window_size = window_size

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.size()
        window_size = self.window_size

        # Pad the sequence to ensure it can be divided into windows
        pad_size = (window_size - (seq_len % window_size)) % window_size
        x = F.pad(x, (0, 0, 0, pad_size))  # Pad along the sequence dimension
        padded_seq_len = x.size(1)

        # Reshape to [batch_size, num_windows, window_size, embed_dim]
        x = x.view(batch_size, -1, window_size, embed_dim)

        # Reshape to [batch_size * num_windows, window_size, embed_dim]
        batched_x = x.contiguous().view(batch_size * x.size(1), window_size, embed_dim)

        # Apply Multi-Head Attention within each window
        attn_output, _ = self.mha(batched_x, batched_x, batched_x)  # [batch_size * num_windows, window_size, embed_dim]

        # Reshape back to [batch_size, num_windows, window_size, embed_dim]
        attn_output = attn_output.view(batch_size, -1, window_size, embed_dim)

        # Reshape back to [batch_size, seq_len, embed_dim]
        attn_output = attn_output.contiguous().view(batch_size, padded_seq_len, embed_dim)

        # Remove padding
        if pad_size > 0:
            attn_output = attn_output[:, :seq_len, :]

        return attn_output


class SpatioTemporalLayer(nn.Module):  # 归一化相似度时空层
    def __init__(self, embed_dim):
        super(SpatioTemporalLayer, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, x, spatial_info, temporal_info):
        # x: [batch_size, seq_len, embed_dim]
        # spatial_info: [batch_size, seq_len, embed_dim]
        # temporal_info: [batch_size, seq_len, embed_dim]

        # Compute spatial similarity based on numerical spatial information
        spatial_similarity = self.spatial_similarity(spatial_info)

        # Compute temporal similarity based on temporal information
        temporal_similarity = self.temporal_similarity(temporal_info)

        # print(spatial_similarity.shape)
        # print(temporal_similarity.shape)

        # Combine spatial and temporal similarity
        similarity = spatial_similarity * temporal_similarity

        # Update POI representations with global spatio-temporal context
        x = x + similarity * x  # Element-wise addition with similarity-weighted x

        return x

    def spatial_similarity(self, spatial_info):
        # Calculate pairwise distances between consecutive spatial_info
        spatial_diff = spatial_info[:, 1:] - spatial_info[:, :-1]
        spatial_diff = torch.abs(spatial_diff)

        # Normalize distance to [0, 1]
        max_distance = spatial_diff.max()
        spatial_similarity = 1 - (spatial_diff / max_distance)

        # Pad the similarity tensor to match the input sequence length
        spatial_similarity = F.pad(spatial_similarity, (0, 0, 1, 0), value=1.0)

        return spatial_similarity

    def temporal_similarity(self, temporal_info):
        # Calculate pairwise time differences between consecutive temporal_info
        time_diff = temporal_info[:, 1:] - temporal_info[:, :-1]
        time_diff = torch.abs(time_diff)

        # Normalize time difference to [0, 1]
        max_time_diff = time_diff.max()
        temporal_similarity = 1 - (time_diff / max_time_diff)

        # Pad the similarity tensor to match the input sequence length
        temporal_similarity = F.pad(temporal_similarity, (0, 0, 1, 0), value=1.0)

        return temporal_similarity


class SpatioTemporalLayer1(nn.Module):  # 显式建模时空信息交互
    def __init__(self, config):
        super(SpatioTemporalLayer1, self).__init__()
        self.config = config
        self.embed_dim = config.Embedding.base_dim
        self.spatial_embedding = nn.Linear(self.embed_dim, self.embed_dim)
        self.temporal_embedding = nn.Linear(self.embed_dim, self.embed_dim)
        self.interaction_layer = nn.MultiheadAttention(self.embed_dim, num_heads=8)

    def forward(self, x, spatial_info, temporal_info):
        # x: [batch_size, seq_len, embed_dim]
        # spatial_info: [batch_size, seq_len, embed_dim]
        # temporal_info: [batch_size, seq_len, embed_dim]

        # Embed spatial and temporal information
        spatial_embed = self.spatial_embedding(spatial_info)
        temporal_embed = self.temporal_embedding(temporal_info)

        # Compute spatial similarity using embedded spatial information
        spatial_similarity = F.cosine_similarity(spatial_embed.unsqueeze(1), spatial_embed.unsqueeze(2), dim=-1)
        spatial_similarity = spatial_similarity.unsqueeze(-1)  # [batch_size, seq_len, seq_len, 1]

        # Compute temporal similarity using embedded temporal information
        temporal_similarity = F.cosine_similarity(temporal_embed.unsqueeze(1), temporal_embed.unsqueeze(2), dim=-1)
        temporal_similarity = temporal_similarity.unsqueeze(-1)  # [batch_size, seq_len, seq_len, 1]

        # Compute interaction between spatial and temporal information
        interaction = self.interaction_layer(spatial_embed, temporal_embed, temporal_embed)[0]
        interaction = interaction.unsqueeze(2)  # [batch_size, seq_len, 1, embed_dim]

        # Combine spatial, temporal, and interaction similarity
        similarity = spatial_similarity * temporal_similarity * interaction  # [batch_size, seq_len, seq_len, embed_dim]

        # Aggregate similarity across the sequence dimension
        similarity = similarity.mean(dim=2)  # [batch_size, seq_len, embed_dim]

        # Update POI representations with global spatio-temporal context
        x = x + similarity  # Element-wise addition with similarity-weighted x

        return x


class EnhancedTransformer(nn.Module):
    def __init__(self, config):
        super(EnhancedTransformer, self).__init__()
        self.config = config
        embed_dim = config.Embedding.base_dim

        self.lw_attention = LocalWindowAttention(config)

        # 新增FFN模块
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # 扩展维度
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),  # 恢复维度
            nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.spatial_temporal_layer = SpatioTemporalLayer1(config)  # 使用显示建模时空交互，而非归一化
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, spatial_info, temporal_info):
        # Local Window Attention + FFN
        residual = x
        x = self.lw_attention(x)
        x = self.norm1(x + residual)  # Add & Norm

        # FFN
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)  # Add & Norm

        # Spatio-Temporal Layer
        x = self.spatial_temporal_layer(x, spatial_info, temporal_info)

        # Final processing
        x = self.dropout(x)

        return x


class EnhancedTransformerStack(nn.Module):
    def __init__(self, config):
        super(EnhancedTransformerStack, self).__init__()
        self.num_layers = 2
        self.layers = nn.ModuleList([
            EnhancedTransformer(config)
            for _ in range(2)
        ])

    def forward(self, x, spatial_info, temporal_info):
        # x: [batch_size, seq_len, embed_dim]
        # spatial_info: [batch_size, seq_len, embed_dim]
        # temporal_info: [batch_size, seq_len, embed_dim]

        for layer in self.layers:
            x = layer(x, spatial_info, temporal_info)

        return x


class TransEncoder(nn.Module):  # TransEncoder用来进行消融实验对比
    def __init__(self, config):
        super(TransEncoder, self).__init__()
        self.config = config
        d_model = config.Embedding.base_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=4,
                                                   dim_feedforward=d_model,
                                                   dropout=0.1,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   )

        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=2,
                                             norm=encoder_norm)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, emb, src_mask):
        emb_out = self.encoder(emb, mask=src_mask)

        return emb_out
