import torch
import pytest
from torch_geometric.data import Data

from bioneuralnet.network_embedding.graph_transformer import GraphTransformer, PositionalEncoding, GraphTransformerLayer


def build_toy_graph(num_nodes: int = 4, input_dim: int = 6) -> Data:
    x = torch.randn((num_nodes, input_dim), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_graph_transformer_forward_regression_shape():
    torch.manual_seed(42)
    data = build_toy_graph(num_nodes=5, input_dim=8)
    model = GraphTransformer(input_dim=8, hidden_dim=16, layer_num=2, heads=4, dropout=0.0,
                             final_layer="regression", activation="gelu", seed=42)
    out = model(data)
    assert out.shape == (data.num_nodes, 1), "Regression head should output [num_nodes, 1]"


def test_graph_transformer_embeddings_shape():
    torch.manual_seed(42)
    data = build_toy_graph(num_nodes=5, input_dim=8)
    hidden_dim = 16
    model = GraphTransformer(input_dim=8, hidden_dim=hidden_dim, layer_num=2, heads=4, dropout=0.0,
                             final_layer="identity", activation="gelu", seed=42)
    emb = model.get_embeddings(data)
    assert emb.shape == (data.num_nodes, hidden_dim), "Embeddings should be [num_nodes, hidden_dim]"


def test_graph_transformer_no_nans_in_output():
    torch.manual_seed(0)
    data = build_toy_graph(num_nodes=6, input_dim=10)
    model = GraphTransformer(input_dim=10, hidden_dim=12, layer_num=3, heads=3, dropout=0.1,
                             final_layer="regression", activation="relu", seed=0)
    out = model(data)
    assert torch.isfinite(out).all(), "Model output should not contain NaNs or Infs"


def test_graph_transformer_backward():
    torch.manual_seed(1)
    data = build_toy_graph(num_nodes=4, input_dim=6)
    model = GraphTransformer(input_dim=6, hidden_dim=8, layer_num=2, heads=2, dropout=0.0,
                             final_layer="regression", activation="gelu", seed=1)
    out = model(data)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads), "Gradients should propagate"


def test_positional_encoding_forward_shape_and_grad():
    torch.manual_seed(7)
    x = torch.randn((10, 16), dtype=torch.float32, requires_grad=True)
    pos_enc = PositionalEncoding(dim=16)
    y = pos_enc(x)
    assert y.shape == x.shape, "PositionalEncoding should preserve shape"
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "PositionalEncoding should be differentiable"


def test_graph_transformer_layer_heads_divisible_assertion():
    with pytest.raises(AssertionError):
        _ = GraphTransformerLayer(in_channels=8, out_channels=10, heads=3, dropout=0.0)
