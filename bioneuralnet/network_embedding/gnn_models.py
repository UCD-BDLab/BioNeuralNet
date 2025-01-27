import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


class GCN(nn.Module):
    """
    GCN with optional final regression layer.
    """

    def __init__(
        self, input_dim, hidden_dim, layer_num=2, dropout=True, final_layer="regression"
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(layer_num - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.final_layer = final_layer

        if self.final_layer == "regression":
            self.regressor = nn.Linear(hidden_dim, 1)
        else:
            self.regressor = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)

        if self.final_layer == "regression" and self.regressor is not None:
            x = self.regressor(x)
        return x

    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class GAT(nn.Module):
    """
    GAT with optional final regression layer.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_num=2,
        dropout=True,
        heads=1,
        final_layer="regression",
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads))
        for _ in range(layer_num - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.dropout = dropout
        self.final_layer = final_layer
        self.heads = heads

        if self.final_layer == "regression":
            self.regressor = nn.Linear(hidden_dim * heads, 1)
        else:
            self.regressor = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)

        if self.final_layer == "regression" and self.regressor is not None:
            x = self.regressor(x)
        return x

    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class SAGE(nn.Module):
    """
    GraphSAGE with optional final regression layer.
    """

    def __init__(
        self, input_dim, hidden_dim, layer_num=2, dropout=True, final_layer="regression"
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(layer_num - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.final_layer = final_layer

        if self.final_layer == "regression":
            self.regressor = nn.Linear(hidden_dim, 1)
        else:
            self.regressor = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)

        if self.final_layer == "regression" and self.regressor is not None:
            x = self.regressor(x)
        return x

    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class GIN(nn.Module):
    """
    GIN with optional final regression layer.
    """

    def __init__(
        self, input_dim, hidden_dim, layer_num=2, dropout=True, final_layer="regression"
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(layer_num):
            nn_module = nn.Sequential(
                nn.Linear(hidden_dim if i > 0 else input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(nn_module))

        self.dropout = dropout
        self.final_layer = final_layer
        if self.final_layer == "regression":
            self.regressor = nn.Linear(hidden_dim, 1)
        else:
            self.regressor = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)

        if self.final_layer == "regression" and self.regressor is not None:
            x = self.regressor(x)
        return x

    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        return x
