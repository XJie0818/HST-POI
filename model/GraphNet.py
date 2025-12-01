import torch
from torch import nn


class GraphNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphNet, self).__init__() # 初始化

        hidden_dim = input_dim * 2
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入维度扩展到两倍
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),  # 将维度缩减回原始维度
        )

        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        out = self.ffn(x) + x  # 残差
        return self.head(out)


class PredictLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PredictLayer, self).__init__()

        hidden_dim = input_dim * 2
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(0.1),
        )

        self.norm = nn.BatchNorm1d(input_dim)
        self.drop = nn.Dropout(0.1)
        num_locations = output_dim
        self.classifier = nn.Linear(input_dim, num_locations)

    def forward(self, x):
        out = self.ffn(x) + x
        out = self.norm(out)  # BN 放在残差后
        out = self.drop(out)
        return self.classifier(out)

