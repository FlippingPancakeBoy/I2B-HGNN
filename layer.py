from torch_geometric.nn import GCNConv, ChebConv, GATConv

from STN import HG_Attention
from STN_IB import HG_Attention_IB
from TRGCN_larger import TransConv, GraphConv, IB_GPool
from TRGCN_small import GCN, TransConv_small
from utils import dataloader
from edge import EDGE, pearson_correlation
from opt import *
from torch import nn
import torch
import torch.nn.functional as F

opt = OptInit().initialize()


class ROI_SATP_GNN(nn.Module):
    def __init__(
            self, args,
            in_channels,
            hidden_channels,
            out_channels,
            trans_num_layers=1,
            trans_num_heads=1,
            trans_dropout=0.5,
            gnn_num_layers=1,
            gnn_dropout=0.5,
            gnn_use_weight=True,
            gnn_use_init=False,
            gnn_use_bn=True,
            gnn_use_residual=True,
            gnn_use_act=True,
            alpha=0.5,
            trans_use_bn=True,
            trans_use_residual=True,
            trans_use_weight=True,
            trans_use_act=True,
            use_graph=True,
            graph_weight=0.8,
            aggregate="add",
            pool_out_channels=64,
            beta=0.8,  
    ):
        super(ROI_SATP_GNN, self).__init__()
        self.args = args
        self.trans_conv = TransConv(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            alpha,
            trans_dropout,
            trans_use_bn,
            trans_use_residual,
            trans_use_weight,
            trans_use_act,
        )
        self.graph_conv = GraphConv(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
            gnn_use_bn,
            gnn_use_residual,
            gnn_use_weight,
            gnn_use_init,
            gnn_use_act,
        )
        self.ib_gpool = IB_GPool(
            in_channels=hidden_channels, 
            out_channels=pool_out_channels,  
            beta=beta, 
        )
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == "add":
            self.fc = nn.Linear(pool_out_channels, out_channels)
        elif aggregate == "cat":
            self.fc = nn.Linear(2 * pool_out_channels, out_channels)
        else:
            raise ValueError(f"Invalid aggregate type:{aggregate}")

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = (
            list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        )
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, edge_index):
        x1 = self.trans_conv(x)
        if self.use_graph:
            x2 = self.graph_conv(x, edge_index)
            if self.aggregate == "add":
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        x_pool, mi_loss = self.ib_gpool(x, edge_index)
        x = self.fc(x_pool)
        x = x
        x = x.view(1, -1)
        return x, mi_loss

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()


class People_GNN(nn.Module):
    def __init__(
            self, args,
            in_channels,
            hidden_channels,
            out_channels,
            trans_num_layers=1,
            trans_num_heads=1,
            trans_dropout=0.5,
            gnn_num_layers=1,
            gnn_dropout=0.5,
            gnn_use_weight=True,
            gnn_use_init=False,
            gnn_use_bn=True,
            gnn_use_residual=True,
            gnn_use_act=True,
            alpha=0.5,
            trans_use_bn=True,
            trans_use_residual=True,
            trans_use_weight=True,
            trans_use_act=True,
            use_graph=True,
            graph_weight=0.8,
            aggregate="add",
    ):
        super(People_GNN, self).__init__()
        self.args = args
        self.trans_conv = TransConv(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            alpha,
            trans_dropout,
            trans_use_bn,
            trans_use_residual,
            trans_use_weight,
            trans_use_act,
        )
        self.graph_conv = GraphConv(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
            gnn_use_bn,
            gnn_use_residual,
            gnn_use_weight,
            gnn_use_init,
            gnn_use_act,
        )
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == "add":
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == "cat":
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f"Invalid aggregate type:{aggregate}")

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = (
            list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        )
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, edge_index):
        x1 = self.trans_conv(x)
        if self.use_graph:
            x2 = self.graph_conv(x, edge_index)
            if self.aggregate == "add":
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        x = self.fc(x)
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x) 
        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()


class H_gnn(torch.nn.Module):

    def __init__(self, args, nonimg, phonetic_score):
        super(H_gnn, self).__init__()
        self.args = args
        self.nonimg = nonimg
        self.phonetic_score = phonetic_score
        self._setup()
        # self.metapath_attention = HG_Attention(node_feature_size=20, node_size=871, num_meta_paths=4, beta=0.1, lambda_wl=0.1, gamma=0.1)
        self.metapath_attention = HG_Attention_IB(node_feature_size=20, node_size=871, num_meta_paths=4,beta=0.1, lambda_wl=0.1, gamma=0.1)

        self.fc1 = nn.Linear(20, 20)
        self.bn1 = torch.nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 10)
        self.bn2 = torch.nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 2)

        self.x_orig = None

    def _setup(self):
        self.ROI_SATP_GNN = ROI_SATP_GNN(self.args, in_channels=116, hidden_channels=64, out_channels=20)
        self.People_GNN = People_GNN(self.args, in_channels=2320, hidden_channels=64, out_channels=20)

    def forward(self, graphs):
        dl = dataloader(self.args)
        embeddings = []
        h = 0
        perms = []
        # 存储第一个图的原始特征作为x_orig
        self.x_orig = graphs[0].x.to(self.args.device)

        # 使用Local ROI-GNN得到每个被试的人群图结点嵌入
        for graph in graphs:
            # 得到graph中的edge_index
            edge_index = graph.edge_index.to(self.args.device)
            x = graph.x.to(self.args.device)
            embedding, mi_loss = self.ROI_SATP_GNN(x, edge_index)
            embeddings.append(embedding)  # 在行维度上添加
        # 把embeddings从列表转换为tensor
        embeddings = torch.cat(tuple(embeddings))

        # 获得四个同构图边索引
        edge_index_site, edge_index_sex, edge_index_age, edge_index_handedness = dl.get_inputs(self.nonimg, embeddings,
                                                                                               self.phonetic_score)

        edge_index_site = torch.tensor(edge_index_site, dtype=torch.long, device=opt.device)
        edge_index_sex = torch.tensor(edge_index_sex, dtype=torch.long, device=opt.device)
        edge_index_age = torch.tensor(edge_index_age, dtype=torch.long, device=opt.device)
        edge_index_handedness = torch.tensor(edge_index_handedness, dtype=torch.long, device=opt.device)

        embeddings = embeddings.to(opt.device)

        emb_site = self.People_GNN(embeddings, edge_index_site)
        emb_sex = self.People_GNN(embeddings, edge_index_sex)
        emb_age = self.People_GNN(embeddings, edge_index_age)
        emb_handedness = self.People_GNN(embeddings, edge_index_handedness)

        subgraph_list = [edge_index_site, edge_index_sex, edge_index_age, edge_index_handedness]
        meta_embs = [emb_site, emb_sex, emb_age, emb_handedness]

        embs, hg_loss = self.metapath_attention(meta_embs, subgraph_list, 871, self.x_orig)
        embs = self.bn1(F.relu(self.fc1(embs)))
        embs = F.dropout(embs, p=self.args.dropout, training=self.training)
        embs = self.bn2(F.relu(self.fc2(embs)))
        embs = F.dropout(embs, p=self.args.dropout, training=self.training)
        predictions = self.fc3(embs)

        return predictions, mi_loss, hg_loss
