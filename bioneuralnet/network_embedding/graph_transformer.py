import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax


class GraphTransformerLayer(MessagePassing):
    """
    Graph Transformer Layer implementation.

    Implements a transformer-style message passing mechanism for graphs with multi-head
    self-attention over neighbors. Each node attends to its incoming neighbors using
    learned query/key/value projections, with optional edge features incorporated into
    attention scores. The output per node is normalized, passed through a feed-forward
    network, and combined via residual connections.


    Args:
      in_channels (int): Input feature dimensionality per node.
      out_channels (int): Output feature dimensionality per node. Must be divisible by ``heads``.
      heads (int): Number of attention heads.
      dropout (float): Dropout probability applied to attention weights and FFN.
      edge_dim (int, optional): Dimensionality of edge features if ``use_edge_features`` is True.
      use_bias (bool): Whether to use bias terms in linear projections.
      use_edge_features (bool): If True, include ``edge_attr`` in attention computation.


    Shapes:
      - x: ``[num_nodes, in_channels]``
      - edge_index: ``[2, num_edges]``
      - edge_attr (optional): ``[num_edges, edge_dim]``
      - output: ``[num_nodes, out_channels]``


    Notes:
      - Attention scores are computed per head and softmax-normalized per destination node.
      - Residual connections are applied around attention and FFN sublayers.
      - Layer normalization is used before attention and FFN.
    """
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, edge_dim=None,
                 use_bias=True, use_edge_features=False):
        super(GraphTransformerLayer, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        assert self.out_channels % self.heads == 0, "out_channels must be divisible by heads"

        self.q_proj = nn.Linear(in_channels, out_channels, bias=use_bias)
        self.k_proj = nn.Linear(in_channels, out_channels, bias=use_bias)
        self.v_proj = nn.Linear(in_channels, out_channels, bias=use_bias)

        self.o_proj = nn.Linear(out_channels, out_channels, bias=use_bias)

        if use_edge_features and edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, heads)

        self.pos_enc = PositionalEncoding(out_channels)

        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.layer_norm2 = nn.LayerNorm(out_channels)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(dropout)
        )

        # cache for interpretability
        self._last_attention = None
        self._last_index = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if hasattr(self, 'edge_proj'):
            nn.init.xavier_uniform_(self.edge_proj.weight)

        # Resetting FFN parameters
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        """
        Forward pass of the GraphTransformerLayer.


        Args:
          x (torch.Tensor): Node features of shape ``[num_nodes, in_channels]``.
          edge_index (torch.LongTensor): Edge indices of shape ``[2, num_edges]``.
          edge_attr (torch.Tensor, optional): Edge features of shape ``[num_edges, edge_dim]``.
          return_attention_weights (bool): Unused in this implementation; kept for API parity.


        Returns:
          torch.Tensor: Updated node features of shape ``[num_nodes, out_channels]``.
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Cache the exact edge_index (including self-loops) used for attention computation
        try:
            self._last_edge_index = edge_index.detach().cpu()
        except Exception:
            self._last_edge_index = None

        x_norm = self.layer_norm1(x)
        # Integrate positional encoding with degree bias and add small training-time noise to break symmetry
        x_with_pos = self.pos_enc(x_norm, edge_index=edge_index)
        if self.training:
            x_with_pos = x_with_pos + 1e-3 * torch.randn_like(x_with_pos)

        attn_out = self.propagate(edge_index, x=x_with_pos, edge_attr=edge_attr,
                                  return_attention_weights=return_attention_weights)

        x = x + attn_out

        x = x + self.ffn(self.layer_norm2(x))

        return x

    def message(self, x_i, x_j, edge_attr=None, edge_index_i=None, index=None, ptr=None, size_i=None):
        """
        Compose messages from source nodes j to destination nodes i.


        Applies multi-head attention using projected queries (on i) and keys/values (on j),
        optionally adding edge-derived attention biases per head.


        Returns:
          torch.Tensor: Per-edge messages aggregated later, shaped ``[num_edges, heads * head_dim]``.
        """
        head_dim = max(1, self.out_channels // self.heads)
        query = self.q_proj(x_i).view(-1, self.heads, head_dim)
        key = self.k_proj(x_j).view(-1, self.heads, head_dim)
        value = self.v_proj(x_j).view(-1, self.heads, head_dim)

        scale = math.sqrt(head_dim)
        attention = (query * key).sum(dim=-1) / (scale if scale > 0 else 1.0)
        # Clamping and sanitizing to avoid NaNs/Infs before softmax
        attention = torch.clamp(attention, min=-50.0, max=50.0)
        attention = torch.nan_to_num(attention, nan=0.0, posinf=50.0, neginf=-50.0)

        # Adding edge features to attention if available
        if self.use_edge_features and edge_attr is not None:
            edge_features = self.edge_proj(edge_attr).view(-1, self.heads)
            attention = attention + edge_features

        # Applying softmax to get attention weights per head (segment-wise by destination node)
        E = attention.size(0)
        h = self.heads
        alpha_heads = []
        idx_e = index if index is not None else edge_index_i
        num_nodes = size_i if size_i is not None else (int(idx_e.max().item()) + 1 if idx_e.numel() > 0 else 0)
        for head in range(h):
            alpha_h = softmax(attention[:, head], index=idx_e, num_nodes=num_nodes)
            alpha_heads.append(alpha_h)
        alpha = torch.stack(alpha_heads, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # cache attention weights for interpretability (detach to avoid grads)
        try:
            self._last_attention = alpha.detach().cpu()
            # destination node indices for segment-wise softmax
            self._last_index = (idx_e.detach().cpu() if idx_e is not None else None)
        except Exception:
            # best-effort caching; ignore if not available
            self._last_attention = None
            self._last_index = None


        out = value * alpha.unsqueeze(-1)


        return out.view(-1, self.heads * head_dim)

    def update(self, aggr_out):
        """
        Final per-node update after aggregation. Applies output projection.
        """
        return self.o_proj(aggr_out)




class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for graph nodes.


    Currently applies a linear projection to the node features to produce a
    positional component that is added to the input features.


    Args:
        dim (int): Feature dimension of the node embeddings.


    Attributes:
        dim (int): Stored feature dimension.
        proj (nn.Linear): Linear projection used to produce the learnable
            positional component.
        deg_proj (nn.Linear): Linear projection mapping scalar node degree to feature space.


    Notes:
        A Laplacian-eigenvector-based positional encoding may be added in the
        future. If enabled, it would compute positional coordinates from the
        (normalized) graph Laplacian per graph in the batch and combine them
        with the learnable projection.
    """
    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
        self.deg_proj = nn.Linear(1, dim)

    def forward(self, x, edge_index=None, batch=None):
        """
        Compute positional encoding for nodes and add it to the input features.


        Args:
            x (torch.Tensor): Node feature matrix of shape ``[num_nodes, dim]``.
            edge_index (torch.LongTensor, optional): Edge index of shape
                ``[2, num_edges]``. If provided, degree-based positional bias is added.
            batch (torch.LongTensor, optional): Batch vector of shape
                ``[num_nodes]``. Unused in the current encoding.


        Returns:
            torch.Tensor: Tensor of shape ``[num_nodes, dim]``, equal to
            ``x + proj(x) + deg_proj(deg)`` when degree is available, else ``x + proj(x)``.
        """
        pos_enc = self.proj(x)
        if edge_index is not None:
            # Compute node degrees and normalize to zero-mean, unit-std
            num_nodes = x.size(0)
            deg = degree(edge_index[0], num_nodes=num_nodes).unsqueeze(-1)
            deg = deg.to(x.dtype).to(x.device)
            deg = (deg - deg.mean()) / (deg.std() + 1e-6)
            pos_enc = pos_enc + self.deg_proj(deg)
        return x + pos_enc




class GraphTransformer(nn.Module):
    """
    Graph Transformer model for node representation learning.

    Stacks multiple GraphTransformerLayer blocks to encode node representations,
    followed by an optional regression head. Supports configurable activation,
    dropout, and number of layers/heads.


    Args:
      input_dim (int): Input node feature dimension.
      hidden_dim (int): Hidden dimension used across layers.
      layer_num (int): Number of transformer layers to stack.
      heads (int): Number of attention heads in each layer.
      dropout (float): Dropout probability.
      final_layer (str): If "regression", adds a scalar-regression head; otherwise identity.
      activation (str): One of {"gelu", "relu", "elu"}.
      seed (int, optional): RNG seed for reproducibility.


    Methods:
      forward(data): Returns model output per node; if regression, shape ``[num_nodes, 1]``.
      get_embeddings(data): Returns hidden embeddings before final head, ``[num_nodes, hidden_dim]``.
    """
    def __init__(self, input_dim, hidden_dim, layer_num=2, heads=4, dropout=0.1,
                 final_layer="regression", activation="gelu", seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        super().__init__()

        self.dropout = dropout
        self.final_layer = final_layer
        self.heads = heads

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(layer_num):
            self.layers.append(GraphTransformerLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout
            ))

        # Output projection
        if final_layer == "regression":
            self.regressor = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, max(4, hidden_dim // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(4, hidden_dim // 2), 1)
            )
        else:
            self.regressor = nn.Identity()

        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Ensure stable initialization across all modules
        self._last_attentions = []
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier init for input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        # Reset child transformer layers
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # Xavier init for regressor linear layers if present
        if isinstance(self.regressor, nn.Sequential):
            for m in self.regressor:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, data):
        """
        Compute per-node outputs given a PyG Data object.


        Args:
          data (torch_geometric.data.Data): Must contain ``x`` and ``edge_index``; optionally ``edge_attr``.


        Returns:
          torch.Tensor: If ``final_layer == 'regression'``, shape ``[num_nodes, 1]``; else ``[num_nodes, hidden_dim]``.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        x = self.input_proj(x)
        x = self.input_norm(x)
        x0 = x  # preserve input diversity with a global skip

        self._last_attentions = []
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # record last attention weights if available
            if hasattr(layer, "_last_attention") and layer._last_attention is not None:
                self._last_attentions.append({
                    "alpha": layer._last_attention,
                    "index": layer._last_index,
                    "edge_index": getattr(layer, "_last_edge_index", None)
                })
            else:
                self._last_attentions.append(None)

        # Add global residual to prevent representational collapse
        x = x + x0

        # Applying final projection
        x = self.regressor(x)

        return x

    def get_last_attentions(self):
        """
        Return cached attention weights per layer from the last forward pass.

        Returns:
          list[dict|None]: For each layer, either None or a dict with keys:
            - 'alpha': tensor of shape [num_edges, heads]
            - 'index': tensor of destination node indices used for segment softmax
        """
        return self._last_attentions

    def get_embeddings(self, data):
        """
        Compute and return node embeddings prior to the final regression/identity head.


        Args:
          data (torch_geometric.data.Data): Graph data with ``x`` and ``edge_index``.


        Returns:
          torch.Tensor: Node embeddings of shape ``[num_nodes, hidden_dim]``.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        x = self.input_proj(x)
        x = self.input_norm(x)
        x0 = x  # preserve input diversity with a global skip

        # Applying transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Add global residual to prevent representational collapse
        x = x + x0

        # Normalize embeddings column-wise to ensure non-degenerate variance
        x = x - x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        x = x / (std + 1e-6)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        return x
