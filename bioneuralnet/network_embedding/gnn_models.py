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

def process_dropout(dropout):
    if isinstance(dropout, bool):
        return 0.5 if dropout else 0.0
    elif isinstance(dropout, float):
        return dropout
    else:
        raise ValueError("Dropout must be either a boolean or a float.")

def get_activation(activation_choice):
    if activation_choice.lower() == "relu":
        return nn.ReLU()
    elif activation_choice.lower() == "elu":
        return nn.ELU()
    elif activation_choice.lower() == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    else:
        raise ValueError(f"Unsupported activation function: {activation_choice}")

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, final_layer="regression", activation="elu", seed=None, self_loop_and_norm=None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

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

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, heads=1, final_layer="regression", activation="elu", seed=None, self_loop_and_norm=None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

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
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, final_layer="regression", activation="elu", seed=None, self_loop_and_norm=None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        super().__init__()

        self.dropout = process_dropout(dropout)
        self.final_layer = final_layer
        self.activation = get_activation(activation)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(layer_num):
            in_dim = input_dim if i == 0 else hidden_dim
            if self_loop_and_norm is not None:
                self.convs.append(SAGEConv(in_dim, hidden_dim,normalize=False))
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
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, final_layer="regression", activation="relu", seed=None, self_loop_and_norm=None):

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        super().__init__()

        self.dropout = process_dropout(dropout)
        self.final_layer = final_layer
        self.activation = get_activation(activation)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(layer_num):
            in_dim = input_dim if i == 0 else hidden_dim
            # at each GIN layer we create an mlp
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
