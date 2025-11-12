# GraphTransformer for BioNeuralNet

This document provides a focused overview and usage guide for the GraphTransformer model integrated into BioNeuralNet, along with a map of the Python files changed as part of this effort.

## Overview

GraphTransformer is a transformer-style architecture for graph-structured data that leverages attention over node features and optional positional encodings to capture structural information.

Key highlights:
- Transformer-style multi-head attention for graphs
- Flexible positional encoding (currently learnable; Laplacian eigenvectors planned)
- Drop-in usage via `GNNEmbedding(model_type='GRAPHTRANSFORMER')`

## Positional Encoding

Current behavior:
- Learnable positional vectors are added to node embeddings inside `PositionalEncoding`.

Planned enhancement:
- Laplacian eigenvector positional encoding (compute per-graph normalized Laplacian eigenvectors; project via the existing linear layer; exclude trivial components; handle batching per graph).

## Quick Start with GNNEmbedding

```python
from bioneuralnet.network_embedding.gnn_embedding import GNNEmbedding

# Example: create an embedding with the GraphTransformer model
embedder = GNNEmbedding(
    model_type='GRAPHTRANSFORMER',
    hidden_dim=128,
    num_layers=3,
    heads=4,
)

# Depending on your pipeline, call fit/transform or use your dataset loader
# embeddings = embedder.fit_transform(graph_data)
```

## Direct Usage with PyTorch Geometric

```python
import torch
from torch_geometric.data import Data
from bioneuralnet.network_embedding.graph_transformer import GraphTransformer

# Create a small synthetic graph
e = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # edge_index
x = torch.randn(3, 128)  # node features

# Build the PyG Data object
data = Data(x=x, edge_index=e)

# Initialize GraphTransformer
model = GraphTransformer(
    in_channels=128,
    hidden_dim=128,
    num_layers=3,
    heads=4,
    dropout=0.1,
)

model.train()
output = model(data)  # forward pass; returns graph-level or node-level embedding per implementation
```

## Evaluation Script

You can run the example evaluation script to compare GraphTransformer behavior on synthetic or prepared data:
- Module: `examples.graph_transformer_evaluation`

Usage example:

```bash
python -m examples.graph_transformer_evaluation --hidden_dim 128 --num_layers 3 --heads 4
```

## Classification Metrics

If phenotype labels are categorical (e.g., PAM50 subtypes), the evaluation script computes Macro F1 and Macro AUC using a logistic regression classifier on the learned sample embeddings. Result plots are saved to `visualization_results`:

- `graph_transformer_classification_f1.png`
- `graph_transformer_classification_auc.png`

Regression metrics (R², RMSE) remain available and are saved as:

- `graph_transformer_comparison.png`

## Attention Visualization

For interpretability, the GraphTransformer caches the last layer’s attention weights during a forward pass. You can access them via:

```python
atts = model.get_last_attentions()  # list per layer; each dict includes 'alpha'
```

When running `examples.graph_transformer_evaluation`, an attention heatmap (averaged across heads) is generated and saved as:

- `graph_transformer_attention_heatmap.png`

This helps inspect which edges (gene–gene interactions) receive higher attention in the model.

## Tests

Unit tests include forward correctness, shape checks, numeric stability, and gradients:
- `tests/test_graph_transformer.py`

Run all tests:

```bash
pytest -q
```

## API Documentation

Full API is documented via Sphinx (napoleon + autodoc + autosummary). After building docs, refer to:
- Module: `bioneuralnet.network_embedding.graph_transformer`
- Classes: `GraphTransformer`, `GraphTransformerLayer`, `PositionalEncoding`

Build docs locally:

```bash
# From the project root
to pip install -r docs/requirements.txt

# Build HTML docs
sphinx-build -b html docs/source docs/_build/html
```

Open `docs/_build/html/index.html` and navigate to the GNNs section.

## Troubleshooting

- If you see NaNs, try lowering learning rate and/or increasing dropout.
- For large graphs, consider precomputing Laplacian eigenvectors offline to avoid runtime overhead.
- Ensure `torch` and `torch_geometric` versions match those in `requirements.txt` and your CUDA setup.

## Suggested Defaults

- hidden_dim: 128
- num_layers: 3
- heads: 4
- dropout: 0.1

## Additional Modified or Referenced Python Files

- bioneuralnet/network_embedding/gnn_embedding.py
  - Wiring for `model_type='GRAPHTRANSFORMER'` and integration with the embedding pipeline, including shape handling and model initialization nuances.

- bioneuralnet/network_embedding/gnn_models.py
  - Ensured GraphTransformer is imported/registered so the factory can construct it via `model_type`.

- examples/graph_transformer_evaluation.py
  - Script to run/evaluate GraphTransformer on synthetic or prepared datasets.
  - CLI flags like `--hidden_dim`, `--num_layers`, `--heads`; saves results under `visualization_results/` (e.g., `graph_transformer_comparison.png`).

- examples/download_tcga_data.py
  - Utility to fetch TCGA clinical/gene expression and STRING PPI resources used by examples/tests.

- examples/preprocess_tcga_string.py
  - Preprocessing pipeline to merge TCGA and STRING resources; prepares artifacts for evaluation and tutorials.


- examples/test_preprocessing.py
  - Example-level checks validating the preprocessing pipeline consistency.

- examples/test_synthetic_data.py
  - Synthetic data generation and quick validation used for model sanity checks and tutorial demos.
