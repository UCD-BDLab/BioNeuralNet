import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


class GCN(nn.Module):
    """
    GCN:
      - Uses GCNConv layers to transform node features.
      - Final layer is a simple linear that outputs 1 dimension per node (for MSE regression).
      - get_embeddings(data) returns the penultimate layer's node embeddings.
    """

    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(layer_num - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout

        # Final linear: from hidden_dim -> 1
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Forward pass for node-level regression.

        Parameters
        ----------
        data : torch_geometric.data.Data
            PyG Data object with attributes:
            - data.x : node features
            - data.edge_index : edge connectivity

        Returns
        -------
        torch.Tensor
            Shape [num_nodes, 1], a single float per node for regression.
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        out = self.regressor(x)  # shape [num_nodes, 1]
        return out

    def get_embeddings(self, data):
        """
        Returns the penultimate node embeddings (shape [num_nodes, hidden_dim]).
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class GAT(nn.Module):
    """
    GAT:
      - Uses GATConv layers to transform node features.
      - Final layer outputs 1 dimension for regression.
      - get_embeddings(data) returns penultimate embeddings.
    """

    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads))
        for _ in range(layer_num - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.dropout = dropout
        self.regressor = nn.Linear(hidden_dim * heads, 1)

    def forward(self, data):
        """
        Forward pass for node-level regression.

        Returns
        -------
        torch.Tensor
            [num_nodes, 1], single float per node.
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        out = self.regressor(x)
        return out

    def get_embeddings(self, data):
        """
        Return penultimate embeddings: [num_nodes, hidden_dim * heads].
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class SAGE(nn.Module):
    """
    SAGE:
      - Uses SAGEConv layers to transform node features.
      - Final layer outputs 1 dimension for regression.
      - get_embeddings(data) returns penultimate embeddings.
    """

    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(layer_num - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Forward pass for node-level regression.

        Returns
        -------
        torch.Tensor
            [num_nodes, 1]
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        out = self.regressor(x)
        return out

    def get_embeddings(self, data):
        """
        Return penultimate embeddings: [num_nodes, hidden_dim].
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class GIN(nn.Module):
    """
    GIN:
      - Uses GINConv layers.
      - Final layer outputs 1 dimension for regression.
      - get_embeddings(data) returns penultimate embeddings.
    """

    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True):
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
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Forward pass for node-level regression.

        Returns
        -------
        torch.Tensor
            [num_nodes, 1]
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        out = self.regressor(x)
        return out

    def get_embeddings(self, data):
        """
        Return penultimate embeddings: [num_nodes, hidden_dim].
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        return x
