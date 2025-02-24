import torch
from scipy.stats import pearsonr
from torch.nn import Linear as Lin, Sequential as Seq
import torch.nn.functional as F
from torch import nn
import numpy as np


class EDGE(torch.nn.Module):
    """接收两个被试的标准化非影像数据作为输入,计算两个被试的相似度"""
    def __init__(self, input_dim, dropout=0.6):
        super(EDGE, self).__init__()
        hidden1 = 128
        self.parser = nn.Sequential(
            nn.Linear(input_dim, hidden1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden1, bias=True),
            )

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 被试i的标准化非影像数据(ηi)
        x1 = x[:, 0:self.input_dim]
        # 被试j的标准化非影像数据(ηj)
        x2 = x[:, self.input_dim:]
        # MLP(ηi)
        h1 = self.parser(x1)
        # MLP(ηj)
        h2 = self.parser(x2)
        # Wij = (Sim(MLP(ηi), MLP(ηj)) + 1)/2
        p = (self.cos(h1, h2) + 1)*0.5
        # 每个元素 p[i][j] 表示第 i 个被试和第 j 个被试之间的相似度
        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)  
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True


def pearson_correlation(edge_index, embedding):
    correlation_coefficients = []
    for i in range(edge_index.shape[1]):
        node1, node2 = edge_index[:, i]
        emb1, emb2 = embedding[node1].cpu().detach().numpy(), embedding[node2].cpu().detach().numpy()  # 将张量转换为NumPy数组
        correlation = np.corrcoef(emb1, emb2)[0, 1]
        correlation_coefficients.append(correlation)
    return correlation_coefficients


