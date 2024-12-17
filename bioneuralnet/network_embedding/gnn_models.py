import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(layer_num - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, heads=1):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads))
        for _ in range(layer_num - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x

class SAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True):
        super(SAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(layer_num - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True):
        super(GIN, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(layer_num):
            nn_module = nn.Sequential(
                nn.Linear(hidden_dim if i > 0 else input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_module))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x
