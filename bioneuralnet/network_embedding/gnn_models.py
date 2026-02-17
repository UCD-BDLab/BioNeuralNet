try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GINEConv
except ModuleNotFoundError:
    raise ImportError(
        "This module requires PyTorch and PyTorch Geometric. "
        "Please install it via: https://bioneuralnet.readthedocs.io/en/latest/installation.html"
    )

from bioneuralnet.utils import set_seed

def process_dropout(dropout):
    """
    Convert dropout input into a valid float probability.

    Args:

        dropout (Union[bool, int, float]): Input dropout specification.

    Returns:

        float: The validated dropout probability.
    """
    if isinstance(dropout, bool):
        return 0.5 if dropout else 0.0
    elif isinstance(dropout, (int, float)):
        return float(dropout)
    else:
        raise ValueError("Dropout must be either a boolean or a float.")

def get_activation(activation_choice):
    """
    Retrieve the corresponding PyTorch activation function based on string name.

    Args:

        activation_choice (str): The name of the activation (relu, elu, leaky_relu).

    Returns:

        nn.Module: The PyTorch activation layer.
    """
    activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
    }
    
    act = activations.get(activation_choice.lower())
    
    if act is None:
        raise ValueError(f"Unsupported activation function: {activation_choice}")
        
    return act

class GCN(nn.Module):
    """
    Graph Convolutional Network

    layer_num=2 -> 1 conv layer (first only, 0 hidden)
    layer_num=4 -> 3 conv layers (first + 2 hidden)

    Args:

        input_dim (int): Dimensionality of input features.
        hidden_dim (int): Dimensionality of hidden layers.
        layer_num (int): Total layer count (including conv_first).
        dropout (Union[bool, float]): Dropout probability or toggle.
        final_layer (str): Head type ("regression" or "none").
        activation (str): Activation function name.
        seed (Optional[int]): Random seed.
        self_loop_and_norm (Optional[bool]): Flags for manual GCNConv normalization.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_num=2,
        dropout=True,
        final_layer="none",
        activation="relu",
        seed=None,
        self_loop_and_norm=None,
        **kwargs,
    ):
        if seed is not None:
            set_seed(seed)
            
        super().__init__()

        self.dropout = process_dropout(dropout)
        self.final_layer = final_layer
        self.activation = get_activation(activation)
        self.layer_num = layer_num

        if self_loop_and_norm is not None:
            self.conv_first = GCNConv(input_dim, hidden_dim, add_self_loops=False, normalize=False)
        else:
            self.conv_first = GCNConv(input_dim, hidden_dim)

        self.conv_hidden = nn.ModuleList()
        for _ in range(max(0, layer_num - 2)):
            if self_loop_and_norm is not None:
                self.conv_hidden.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False))
            else:
                self.conv_hidden.append(GCNConv(hidden_dim, hidden_dim))

        self.regressor = nn.Linear(hidden_dim, 1) if final_layer == "regression" else nn.Identity()

    def _message_pass(self, data):
        """
        Internal execution of the graph convolutional layers.
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_attr", None)

        x = self.conv_first(x, edge_index, edge_weight=edge_weight)
        x = self.activation(x)
        
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.conv_hidden:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return x

    def forward(self, data):
        """
        Full forward pass including the task-specific head.
        """
        x = self._message_pass(data)
        
        return self.regressor(x)

    def get_embeddings(self, data):
        """
        Extract latent node embeddings.
        """
        return self._message_pass(data)

class GAT(nn.Module):
    """
    Graph Attention Network - uses edge_dim=1 to incorporate edge weights.

    In DPMON edge_dim=1 in GATConv so the attention mechanism can leverage the network's structural information.

    Args:

        input_dim (int): Dimensionality of input features.
        hidden_dim (int): Dimensionality of hidden layers.
        layer_num (int): Total layer count.
        heads (int): Number of attention heads.
        dropout (Union[bool, float]): Dropout probability.
        final_layer (str): Head type.
        activation (str): Activation function name.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_num=2,
        dropout=True,
        heads=1,
        final_layer="none",
        activation="relu",
        seed=None,
        self_loop_and_norm=None,
        **kwargs,
    ):
        if seed is not None:
            set_seed(seed)
            
        super().__init__()

        self.dropout = process_dropout(dropout)
        self.final_layer = final_layer
        self.heads = heads
        self.activation = get_activation(activation)
        self.layer_num = layer_num

        if self_loop_and_norm is not None:
            self.conv_first = GATConv(input_dim, hidden_dim, heads=heads, edge_dim=1, add_self_loops=False)
        else:
            self.conv_first = GATConv(input_dim, hidden_dim, heads=heads, edge_dim=1)

        self.conv_hidden = nn.ModuleList()
        for _ in range(max(0, layer_num - 2)):
            in_dim = hidden_dim * heads
            if self_loop_and_norm is not None:
                self.conv_hidden.append(GATConv(in_dim, hidden_dim, heads=heads, edge_dim=1, add_self_loops=False))
            else:
                self.conv_hidden.append(GATConv(in_dim, hidden_dim, heads=heads, edge_dim=1))

        out_dim = hidden_dim * heads
        self.regressor = nn.Linear(out_dim, 1) if final_layer == "regression" else nn.Identity()

    def _message_pass(self, data):
        """
        Internal execution of the graph attention layers.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        x = self.conv_first(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.conv_hidden:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return x

    def forward(self, data):
        """
        Full forward pass.
        """
        x = self._message_pass(data)
        
        return self.regressor(x)

    def get_embeddings(self, data):
        """
        Extract latent node embeddings.
        """
        return self._message_pass(data)

class SAGE(nn.Module):
    """
    GraphSAGE - aligned layer_num convention.

    Note: SAGEConv does not natively support edge weights.

    Args:

        input_dim (int): Dimensionality of input features.
        hidden_dim (int): Dimensionality of hidden layers.
        layer_num (int): Total layer count.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_num=2,
        dropout=True,
        final_layer="none",
        activation="relu",
        seed=None,
        self_loop_and_norm=None,
        **kwargs,
    ):
        if seed is not None:
            set_seed(seed)
            
        super().__init__()

        self.dropout = process_dropout(dropout)
        self.final_layer = final_layer
        self.activation = get_activation(activation)
        self.layer_num = layer_num

        if self_loop_and_norm is not None:
            self.conv_first = SAGEConv(input_dim, hidden_dim, normalize=False)
        else:
            self.conv_first = SAGEConv(input_dim, hidden_dim)

        self.conv_hidden = nn.ModuleList()
        for _ in range(max(0, layer_num - 2)):
            if self_loop_and_norm is not None:
                self.conv_hidden.append(SAGEConv(hidden_dim, hidden_dim, normalize=False))
            else:
                self.conv_hidden.append(SAGEConv(hidden_dim, hidden_dim))

        self.regressor = nn.Linear(hidden_dim, 1) if final_layer == "regression" else nn.Identity()

    def _message_pass(self, data):
        """
        Internal execution of the SAGE layers.
        """
        x, edge_index = data.x, data.edge_index

        x = self.conv_first(x, edge_index)
        x = self.activation(x)
        
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.conv_hidden:
            x = conv(x, edge_index)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return x

    def forward(self, data):
        """
        Full forward pass.
        """
        x = self._message_pass(data)
        
        return self.regressor(x)

    def get_embeddings(self, data):
        """
        Extract latent node embeddings.
        """
        return self._message_pass(data)

class GIN(nn.Module):
    """
    Graph Isomorphism Network - uses GINEConv for edge-weight awareness.

    DPMON utilizes GINEConv with edge_dim=1 to incorporate edge weights into the MLP-based message passing.

    Args:

        input_dim (int): Dimensionality of input features.
        hidden_dim (int): Dimensionality of hidden layers.
        layer_num (int): Total layer count.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_num=2,
        dropout=True,
        final_layer="none",
        activation="relu",
        seed=None,
        self_loop_and_norm=None,
        output_dim=None,
        **kwargs,
    ):
        if seed is not None:
            set_seed(seed)
            
        super().__init__()

        self.dropout = process_dropout(dropout)
        self.final_layer = final_layer
        self.activation = get_activation(activation)
        self.layer_num = layer_num

        first_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv_first = GINEConv(first_nn, edge_dim=1)

        self.conv_hidden = nn.ModuleList()
        for _ in range(max(0, layer_num - 2)):
            hidden_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv_hidden.append(GINEConv(hidden_nn, edge_dim=1))

        self.regressor = nn.Linear(hidden_dim, 1) if final_layer == "regression" else nn.Identity()

    def _message_pass(self, data):
        """
        Internal execution of the GINE layers.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        x = self.conv_first(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.conv_hidden:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return x

    def forward(self, data):
        """
        Full forward pass.
        """
        x = self._message_pass(data)
        
        return self.regressor(x)

    def get_embeddings(self, data):
        """
        Extract latent node embeddings.
        """
        return self._message_pass(data)