from torch_geometric.nn.pool.topk_pool import topk, filter_adj

from torch_geometric.nn.pool.topk_pool import topk

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

# Weisfeiler-Lehman 图同构检测类
class WL_Isomorphism:
    def __init__(self, num_iterations=3):
        self.num_iterations = num_iterations

    def test_isomorphism(self, edge_index1, edge_index2, num_nodes):
        """判断两个子图是否同构，使用WL同构测试"""
        # 将 edge_index 转换为 torch_geometric.data.Data 对象
        G1 = self.edge_index_to_graph(edge_index1, num_nodes)
        G2 = self.edge_index_to_graph(edge_index2, num_nodes)

        # 使用 networkx 判断图同构
        return nx.is_isomorphic(G1, G2, node_match=self.wl_hash_match)

    def wl_hash_match(self, n1, n2):
        """WL同构测试的节点标签匹配函数"""
        return n1['label'] == n2['label']

    def edge_index_to_graph(self, edge_index, num_nodes):
        """将 edge_index 转换为 networkx 图"""
        # 创建包含节点数目的 Data 对象
        data = Data(edge_index=edge_index, num_nodes=num_nodes)

        # 转换为 networkx 图
        return to_networkx(data)



class HG_Attention_IB(nn.Module):
    def __init__(self, node_feature_size, node_size, num_meta_paths, beta=0.1, lambda_wl=0.1, gamma=0.1):
        super(HG_Attention_IB, self).__init__()
        self.node_feature_size = node_feature_size
        self.node_size = node_size
        self.num_meta_paths = num_meta_paths
        self.beta = beta
        self.lambda_wl = lambda_wl
        self.gamma = gamma

        self.x_proj = nn.Linear(116*116, node_feature_size)  # 116是x_orig的特征维度
        self.W = nn.Parameter(torch.Tensor(self.node_size, 1))
        self.b = nn.Parameter(torch.Tensor(self.node_size, 1))
        self.q = nn.Parameter(torch.Tensor(self.node_size, self.node_feature_size))
        
        # 互信息估计器网络
        self.mi_estimator_xz = nn.Sequential(
            nn.Linear(node_feature_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.mi_estimator_zy = nn.Sequential(
            nn.Linear(node_feature_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 初始化参数
        nn.init.xavier_uniform_(self.W)
        nn.init.uniform_(self.b)
        nn.init.xavier_uniform_(self.q)
        
        self.wl_iso = WL_Isomorphism()

    def estimate_MI(self, x, z, estimator):
        """
        x: [1, 116, 116] tensor
        z: [871, z_dim] tensor
        """
        x = x.squeeze(0) 
        # 然后扩展
        x = x.unsqueeze(0).expand(871, -1, -1)
        
        # 将x重塑
        x = x.reshape(871, -1)
        
        # 使用x_proj投影到正确的维度
        x = self.x_proj(x)
        
        batch_size = z.shape[0]
        z_shuffle = z[torch.randperm(batch_size)]
        # 拼接
        t0 = torch.cat([x, z], dim=1) 
        t1 = torch.cat([x, z_shuffle], dim=1) 
        
        mi_lb = (estimator(t0).mean() - 
                torch.log(torch.exp(estimator(t1)).mean() + 1e-8))
        
        return torch.clamp(mi_lb, min=0.0)

    def forward(self, z_list, subgraph_list, num_nodes, x_orig, y_labels=None):
        """
        x_orig: 原始输入特征
        y_labels: 节点标签（如果有的话）
        """
        semantic_scores = []
        mean_semantic_score = []
        
        # 检测同构子图
        isomorphic_pairs = []
        for i in range(self.num_meta_paths):
            for j in range(i + 1, self.num_meta_paths):
                if self.wl_iso.test_isomorphism(subgraph_list[i], subgraph_list[j], num_nodes):
                    isomorphic_pairs.append((i, j))

        # 计算注意力权重
        for z in z_list:
            sum1 = 0
            for i in range(z.shape[0]):
                z_transformed = torch.tanh(
                    torch.matmul(self.W[i].view(1, 1), z[i].view(1, self.node_feature_size)) + self.b[i].view(1, 1))
                semantic_score = torch.matmul(z_transformed, self.q[i].view(self.node_feature_size, 1))
                sum1 = sum1 + semantic_score
            mean = sum1 / (z.shape[0])
            mean_semantic_score.append(mean)

        combined_tensor = torch.stack(mean_semantic_score).squeeze()
        total_sum = torch.sum(combined_tensor)
        proportions = combined_tensor / total_sum
        attention_weights = F.softmax(proportions, dim=0)

        # 处理同构路径的注意力权重
        for (i, j) in isomorphic_pairs:
            attention_weights[j] = attention_weights[i]

        # 应用注意力权重
        layer_out = []
        for i in range(len(z_list)):
            a = z_list[i]
            z_list[i] = attention_weights[i] * z_list[i]
            layer_out.append(z_list[i] + a)  # 残差连接

        # 最终嵌入
        emb = sum(layer_out)
        # 确保x_orig在正确的设备上
        x_orig = x_orig.to(emb.device)
        
        # 计算信息瓶颈损失
        I_XZ = self.estimate_MI(x_orig, emb, self.mi_estimator_xz)
        if y_labels is not None:
            y_onehot = F.one_hot(y_labels, num_classes=y_labels.max()+1).float()
            I_ZY = self.estimate_MI(emb, y_onehot, self.mi_estimator_zy)
        else:
            I_ZY = torch.tensor(0.0).to(emb.device)

        # 信息瓶颈损失
        ib_loss = I_XZ - self.beta * I_ZY
        
        # WL同构损失
        wl_loss = torch.sum(torch.tensor([1.0 if (i, j) in isomorphic_pairs else 0.0 
                           for i in range(self.num_meta_paths) 
                           for j in range(i + 1, self.num_meta_paths)]))

        # 注意力正则化
        attention_reg = self.gamma * torch.sum(attention_weights)

        # 总损失
        total_loss = ib_loss + self.lambda_wl * wl_loss + attention_reg

        return emb, total_loss