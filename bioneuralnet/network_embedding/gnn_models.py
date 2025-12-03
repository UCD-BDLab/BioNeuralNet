try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
except ModuleNotFoundError:
    raise ImportError(
        "This module requires PyTorch and PyTorch Geometric. "
        "Please install it by following the instructions at: "
        "https://bioneuralnet.readthedocs.io/en/latest/installation.html"
    )

from bioneuralnet.utils import set_seed

def process_dropout(dropout):
    """Convert dropout input into a valid float probability."""
    if isinstance(dropout, bool):
        return 0.5 if dropout else 0.0
    elif isinstance(dropout, float):
        return dropout
    else:
        raise ValueError("Dropout must be either a boolean or a float.")

def get_activation(activation_choice):
    """Retrieve the corresponding PyTorch activation function based on string name."""
    if activation_choice.lower() == "relu":
        return nn.ReLU()
    elif activation_choice.lower() == "elu":
        return nn.ELU()
    elif activation_choice.lower() == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    else:
        raise ValueError(f"Unsupported activation function: {activation_choice}")

class GCN(nn.Module):
    """Graph Convolutional Network implementation.

    Args:

        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden layers.
        layer_num (int): Number of GCN layers.
        dropout (bool | float): Dropout probability.
        final_layer (str): Type of final layer ('regression' or identity).
        activation (str): Activation function name.
        seed (int | None): Random seed for reproducibility.
        self_loop_and_norm (bool | None): Whether to add self-loops and normalize.

    """
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, final_layer="regression", activation="relu", seed=None, self_loop_and_norm=None):
        if seed is not None:
            set_seed(seed)

        super().__init__()
        self.dropout = process_dropout(dropout)
        self.final_layer = final_layer
        self.activation = get_activation(activation)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(layer_num):
            in_dim = input_dim if i == 0 else hidden_dim
            if self_loop_and_norm is not None:
                self.convs.append(GCNConv(in_dim, hidden_dim, add_self_loops=False, normalize=False))
            else:
                self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.Identity())

        self.regressor = nn.Linear(hidden_dim, 1) if self.final_layer == "regression" else nn.Identity()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.regressor(x)
        return x

    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GAT(nn.Module):
    """Graph Attention Network implementation.

    Args:

        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden layers.
        layer_num (int): Number of GAT layers.
        dropout (bool | float): Dropout probability.
        heads (int): Number of attention heads.
        final_layer (str): Type of final layer ('regression' or identity).
        activation (str): Activation function name.
        seed (int | None): Random seed for reproducibility.
        self_loop_and_norm (bool | None): Whether to add self-loops.

    """
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, heads=1, final_layer="regression", activation="relu", seed=None, self_loop_and_norm=None):
        if seed is not None:
            set_seed(seed)

        super().__init__()

        self.dropout = process_dropout(dropout)
        self.final_layer = final_layer
        self.heads = heads
        self.activation = get_activation(activation)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(layer_num):
            in_dim = input_dim if i == 0 else hidden_dim * heads
            if self_loop_and_norm is not None:
                self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, add_self_loops=False))
            else:
                self.convs.append(GATConv(in_dim, hidden_dim, heads=heads))
            self.bns.append(nn.Identity())

        self.regressor = nn.Linear(hidden_dim * heads, 1) if self.final_layer == "regression" else nn.Identity()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.regressor(x)
        return x

    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class SAGE(nn.Module):
    """GraphSAGE Network implementation.

    Args:

        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden layers.
        layer_num (int): Number of SAGE layers.
        dropout (bool | float): Dropout probability.
        final_layer (str): Type of final layer ('regression' or identity).
        activation (str): Activation function name.
        seed (int | None): Random seed for reproducibility.
        self_loop_and_norm (bool | None): Whether to normalize.

    """
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, final_layer="regression", activation="relu", seed=None, self_loop_and_norm=None):
        if seed is not None:
            set_seed(seed)

        super().__init__()

        self.dropout = process_dropout(dropout)
        self.final_layer = final_layer
        self.activation = get_activation(activation)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(layer_num):
            in_dim = input_dim if i == 0 else hidden_dim
            if self_loop_and_norm is not None:
                self.convs.append(SAGEConv(in_dim, hidden_dim, normalize=False))
            else:
                self.convs.append(SAGEConv(in_dim, hidden_dim))
            self.bns.append(nn.Identity())

        self.regressor = nn.Linear(hidden_dim, 1) if self.final_layer == "regression" else nn.Identity()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.regressor(x)
        return x

    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GIN(nn.Module):
    """Graph Isomorphism Network (GIN) implementation.

    Args:

        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden layers.
        layer_num (int): Number of GIN layers.
        dropout (bool | float): Dropout probability.
        final_layer (str): Type of final layer ('regression' or identity).
        activation (str): Activation function name.
        seed (int | None): Random seed for reproducibility.
        self_loop_and_norm (bool | None): Unused in GIN, present for API consistency.
        output_dim (int | None): Optional output dimension override.

    """
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, final_layer="regression", activation="relu", seed=None, self_loop_and_norm=None, output_dim=None):
        if seed is not None:
            set_seed(seed)

        super().__init__()

        self.dropout = process_dropout(dropout)
        self.final_layer = final_layer
        self.activation = get_activation(activation)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(layer_num):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.Identity())

        self.regressor = nn.Linear(hidden_dim, 1) if self.final_layer == "regression" else nn.Identity()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.regressor(x)
        return x

    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
